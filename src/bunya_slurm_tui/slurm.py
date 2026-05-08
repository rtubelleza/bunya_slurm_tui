"""base validation classes
:author: Rafael Tubelleza <r.tubelleza@uq.edu.au>
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# TODO: maybe call these from some spec request rather than static constants
BunyaPartition = Literal["general", "gpu_cuda", "gpu_rocm", "gpu_sxm", "gpu_viz"]
BunyaQoS = Literal["normal", "debug", "short", "gpu", "mig", "sxm", "sdf", "viz"]
BunyaArchConstraint = Literal["epyc3", "epyc4"]


_GPU_PARTITIONS: dict[str, set[str]] = {
    "a100": {"gpu_cuda"},
    "nvidia_a100_80gb_pcie_1g.10gb": {"gpu_cuda"},
    "nvidia_a100_80gb_pcie_2g.20gb": {"gpu_cuda"},
    "nvidia_a100_80gb_pcie_3g.40gb": {"gpu_cuda"},
    "h100": {"gpu_cuda", "gpu_sxm"},
    "l40": {"gpu_cuda", "gpu_viz"},
    "l40s": {"gpu_cuda", "gpu_viz"},
    "a16": {"gpu_viz"},
    "mi210": {"gpu_rocm"},
    "mi300x": {"gpu_rocm"},
}

_GPU_MAX_CPUS: dict[tuple[str, str], int] = {
    ("a100", "gpu_cuda"): 86,
    ("nvidia_a100_80gb_pcie_1g.10gb", "gpu_cuda"): 22,
    ("nvidia_a100_80gb_pcie_2g.20gb", "gpu_cuda"): 22,
    ("nvidia_a100_80gb_pcie_3g.40gb", "gpu_cuda"): 22,
    ("h100", "gpu_cuda"): 64,
    ("h100", "gpu_sxm"): 48,
    ("l40", "gpu_cuda"): 64,
    ("l40", "gpu_viz"): 64,
    ("l40s", "gpu_cuda"): 64,
    ("l40s", "gpu_viz"): 64,
    ("a16", "gpu_viz"): 16,
    ("mi210", "gpu_rocm"): 96,
    ("mi300x", "gpu_rocm"): 16,
}

_PARTITION_QOS: dict[str, set[str]] = {
    "general": {"normal", "debug", "short"},
    "gpu_cuda": {"gpu", "debug", "short", "mig"},
    "gpu_rocm": {"gpu", "debug", "short", "sdf"},
    "gpu_sxm": {"sxm"},
    "gpu_viz": {"viz"},
}


def _parse_gres(gres: str) -> tuple[str | None, int]:
    parts = gres.split(":")
    if len(parts) == 3:
        return parts[1], int(parts[2])
    if len(parts) == 2:
        return None, int(parts[1])
    raise ValueError(f"gres must be 'gpu:TYPE:COUNT' or 'gpu:COUNT', got: {gres!r}")


class BunyaSlurmJobSpec(BaseModel):
    job_name: Annotated[str, Field(description="SLURM job name")]
    account: Annotated[str, Field(description="Bunya account (must start with 'a_')")]

    partition: Annotated[BunyaPartition, Field()] = "general"
    qos: Annotated[BunyaQoS, Field()] = "normal"

    nodes: Annotated[int, Field()] = 1
    ntasks_per_node: Annotated[int, Field()] = 1
    cpus_per_task: Annotated[int, Field()] = 1
    mem: Annotated[str, Field(description="e.g. '10G', '500M'")] = "4G"
    walltime: Annotated[str, Field(description="[D-]HH:MM:SS")] = "1:00:00"

    gres: Annotated[str | None, Field(description="e.g. 'gpu:a100:1'")] = None

    output: Annotated[str, Field()] = "slurm-%j.output"
    error: Annotated[str, Field()] = "slurm-%j.error"

    # exactly one of these must be set.
    script: Annotated[str | None, Field(description="Path to a .sh on Bunya")] = None
    command: Annotated[str | None, Field(description="Inline shell command")] = None
    inline_script_path: Annotated[
        str | None, Field(description="Path to a .py — base64-embedded in script")
    ] = None
    inline_script_args: Annotated[str, Field()] = ""

    # default is whatever python`resolves to inside
    # the SLURM job's environment (after source_files run). Override to a
    # venv interpreter or an apptainer-prefixed command, ie
    #/scratch/u/me/.venv/bin/python
    #apptainer exec --nv -B /scratch image.sif python
    python_cmd: Annotated[str, Field(description="Command used to invoke python")] = "python"

    source_files: Annotated[list[str], Field(default_factory=list)] = Field(
        default_factory=list
    )
    # Lmod / environment modules to load before the payload runs. Emits
    # module load (modulues) for each
    modules: Annotated[list[str], Field(default_factory=list)] = Field(
        default_factory=list
    )
    # Env vars to export in the SLURM script body. if 
    # using apptainer, then vars are also forwardewd to apptainer env 
    # APPTAINERENV_<KEY>=<VALUE> so it crosses the container boundary
    env_vars: Annotated[dict[str, str], Field(default_factory=dict)] = Field(
        default_factory=dict
    )
    constraint: Annotated[BunyaArchConstraint | None, Field()] = None
    chdir: Annotated[str | None, Field()] = None
    extra_flags: Annotated[list[str], Field(default_factory=list)] = Field(
        default_factory=list
    )

    @field_validator("account")
    @classmethod
    def _account_prefix(cls, v: str) -> str:
        if not v.startswith("a_"):
            raise ValueError(f"Bunya account must start with 'a_', got: {v!r}")
        return v

    @field_validator("mem")
    @classmethod
    def _mem_format(cls, v: str) -> str:
        if not re.fullmatch(r"\d+[KMGT]?", v, re.IGNORECASE):
            raise ValueError(f"mem must be a number with optional K/M/G/T, got: {v!r}")
        return v.upper()

    @field_validator("walltime")
    @classmethod
    def _walltime_format(cls, v: str) -> str:
        if not re.fullmatch(r"(?:\d+-)?(?:\d+):\d{2}:\d{2}", v):
            raise ValueError(f"walltime must be [D-]HH:MM:SS, got: {v!r}")
        return v

    @field_validator("gres")
    @classmethod
    def _gres_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.startswith("gpu:"):
            raise ValueError(f"gres must start with 'gpu:', got: {v!r}")
        _parse_gres(v)
        return v

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> "BunyaSlurmJobSpec":
        n = sum(
            x is not None for x in (self.script, self.command, self.inline_script_path)
        )
        if n == 0:
            raise ValueError("set one of: script, command, inline_script_path")
        if n > 1:
            raise ValueError(
                "set only one of: script, command, inline_script_path"
            )
        if self.inline_script_path and not Path(self.inline_script_path).exists():
            raise ValueError(
                f"inline_script_path does not exist: {self.inline_script_path!r}"
            )
        return self

    @model_validator(mode="after")
    def _gpu_partition_consistency(self) -> "BunyaSlurmJobSpec":
        if self.gres and self.partition == "general":
            raise ValueError(
                f"gres={self.gres!r} requires a GPU partition, not 'general'"
            )
        if self.partition != "general" and not self.gres:
            raise ValueError(f"partition={self.partition!r} requires a 'gres' value")
        if self.gres:
            gpu_type, gpu_count = _parse_gres(self.gres)
            if gpu_type is not None:
                if gpu_type not in _GPU_PARTITIONS:
                    raise ValueError(
                        f"unknown GPU type {gpu_type!r}; "
                        f"valid: {sorted(_GPU_PARTITIONS)}"
                    )
                if self.partition not in _GPU_PARTITIONS[gpu_type]:
                    raise ValueError(
                        f"GPU {gpu_type!r} not on partition {self.partition!r}; "
                        f"valid: {sorted(_GPU_PARTITIONS[gpu_type])}"
                    )
                max_cpus = _GPU_MAX_CPUS.get((gpu_type, self.partition))
                if max_cpus is not None:
                    total = self.cpus_per_task * self.ntasks_per_node * self.nodes
                    limit = max_cpus * gpu_count
                    if total > limit:
                        raise ValueError(
                            f"{gpu_type!r} on {self.partition!r}: max {max_cpus} CPUs/GPU "
                            f"({gpu_count} GPU(s) -> limit {limit}); requested {total}"
                        )
        return self

    @model_validator(mode="after")
    def _qos_for_partition(self) -> "BunyaSlurmJobSpec":
        valid = _PARTITION_QOS.get(self.partition, set())
        if self.qos not in valid:
            raise ValueError(
                f"qos={self.qos!r} invalid for partition={self.partition!r}; "
                f"valid: {sorted(valid)}"
            )
        return self

    def to_sbatch_flags(self) -> list[str]:
        flags = [
            f"--job-name={self.job_name}",
            f"--account={self.account}",
            f"--partition={self.partition}",
            f"--qos={self.qos}",
            f"--nodes={self.nodes}",
            f"--ntasks-per-node={self.ntasks_per_node}",
            f"--cpus-per-task={self.cpus_per_task}",
            f"--mem={self.mem}",
            f"--time={self.walltime}",
            f"--output={self.output}",
            f"--error={self.error}",
        ]
        if self.gres:
            flags.append(f"--gres={self.gres}")
        if self.constraint:
            flags.append(f"--constraint={self.constraint}")
        if self.chdir:
            flags.append(f"--chdir={self.chdir}")
        flags.extend(self.extra_flags)
        return flags

    def to_script_content(self, extra_env: dict[str, str] | None = None) -> str:
        lines = ["#!/bin/bash --login", ""]
        for flag in self.to_sbatch_flags():
            lines.append(f"#SBATCH {flag}")
        lines.append("#SBATCH --export=ALL")
        lines.append("")
        lines.append("set -eo pipefail")
        lines.append('echo "[slurm-job] starting on $(hostname) at $(date)"')
        lines.append("")

        # Merge spec-level env_vars with any runtime extra_env override.
        # When python_cmd is an apptainer exec, mirror each var as
        # APPTAINERENV_<KEY> so it crosses into the container.
        merged_env = {**self.env_vars, **(extra_env or {})}
        is_apptainer = "apptainer" in self.python_cmd
        if merged_env:
            for k, v in merged_env.items():
                lines.append(f'export {k}="{v}"')
                if is_apptainer:
                    lines.append(f'export APPTAINERENV_{k}="{v}"')
            lines.append("")

        for m in self.modules:
            lines.append(f'echo "[slurm-job] module load {m}"')
            lines.append(f"module load {m}")
        if self.modules:
            lines.append("")

        for f in self.source_files:
            lines.append(f'echo "[slurm-job] sourcing {f}"')
            lines.append(f"source {f}")
        if self.source_files:
            lines.append("")

        if self.inline_script_path:
            encoded = base64.b64encode(
                Path(self.inline_script_path).read_bytes()
            ).decode("ascii")
            name = Path(self.inline_script_path).name
            lines.extend(
                [
                    'INLINE_SCRIPT_DIR="/tmp/${SLURM_JOB_ID}"',
                    'mkdir -p "$INLINE_SCRIPT_DIR"',
                    f'INLINE_SCRIPT="$INLINE_SCRIPT_DIR/{name}"',
                    f'echo "{encoded}" | base64 -d > "$INLINE_SCRIPT"',
                    'echo "[slurm-job] wrote inline script to $INLINE_SCRIPT"',
                    f'{self.python_cmd} "$INLINE_SCRIPT" {self.inline_script_args}',
                ]
            )
        elif self.command:
            lines.append(self.command)
        else:
            lines.append(f"bash {self.script}")

        lines.append("")
        return "\n".join(lines)