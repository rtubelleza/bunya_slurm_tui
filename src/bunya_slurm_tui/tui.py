"""Main TUI app;
:author: Rafael Tubelleza <rafaelrtubelleza@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import questionary
from pydantic import ValidationError
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from bunya_slurm_tui.slurm import (
    _GPU_MAX_CPUS,
    _GPU_PARTITIONS,
    _PARTITION_QOS,
    BunyaSlurmJobSpec,
)

CONSOLE = Console()
#: cache in ~ or home dir previous parameters made by the user
PROFILE_PATH = Path.home() / ".config" / "run-slurm" / "last.json"


def _load_profile() -> dict[str, Any]:
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_profile(spec: BunyaSlurmJobSpec) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    persistable = spec.model_dump()
    for k in ("script", "command", "inline_script_path", "inline_script_args"):
        persistable[k] = None if not isinstance(persistable[k], list) else []
    PROFILE_PATH.write_text(json.dumps(persistable, indent=2))


def _ask_int(prompt: str, default: int, min_value: int = 1) -> int:
    while True:
        raw = questionary.text(
            prompt, default=str(default), validate=lambda x: x.isdigit()
        ).unsafe_ask()
        v = int(raw)
        if v >= min_value:
            return v
        CONSOLE.print(f"[red]must be >= {min_value}[/red]")


def _build_spec(file_path: Path) -> BunyaSlurmJobSpec:
    profile = _load_profile()

    job_name = questionary.text(
        "Job name", default=profile.get("job_name", file_path.stem)
    ).unsafe_ask()

    account_default = (
        profile.get("account") or os.environ.get("SBATCH_ACCOUNT") or "a_"
    )
    account = questionary.text(
        "Bunya account (must start with 'a_')", default=account_default
    ).unsafe_ask()

    # ask GPU first, this is the limiting param which constrains partitions and qols to gpu ones
    want_gpu = questionary.confirm(
        "Use a GPU?",
        default=bool(profile.get("gres")),
    ).unsafe_ask()

    gres: str | None = None
    if want_gpu:
        gpu_types = sorted(_GPU_PARTITIONS.keys())
        last_gpu = (
            profile.get("gres", "").split(":")[1]
            if profile.get("gres") and len(profile.get("gres", "").split(":")) == 3
            else None
        )
        gpu_type = questionary.select(
            "GPU type",
            choices=gpu_types,
            default=last_gpu if last_gpu in gpu_types else gpu_types[0],
        ).unsafe_ask()

        valid_partitions = sorted(_GPU_PARTITIONS[gpu_type])
        partition = (
            valid_partitions[0]
            if len(valid_partitions) == 1
            else questionary.select(
                f"Partition (constrained by GPU={gpu_type})",
                choices=valid_partitions,
            ).unsafe_ask()
        )

        gpu_count = _ask_int("GPU count", default=1)
        gres = f"gpu:{gpu_type}:{gpu_count}"
    else:
        partition = "general"

    qos_choices = sorted(_PARTITION_QOS[partition])
    qos = questionary.select(
        f"QoS (constrained by partition={partition})",
        choices=qos_choices,
        default=(
            profile.get("qos")
            if profile.get("qos") in qos_choices
            else qos_choices[0]
        ),
    ).unsafe_ask()

    nodes = _ask_int("Nodes", default=profile.get("nodes", 1))
    ntasks_per_node = _ask_int(
        "ntasks-per-node", default=profile.get("ntasks_per_node", 1)
    )

    cpus_default = profile.get("cpus_per_task", 1)
    if gres:
        gpu_type, gpu_count = gres.split(":")[1], int(gres.split(":")[2])
        max_per_gpu = _GPU_MAX_CPUS.get((gpu_type, partition))
        if max_per_gpu:
            CONSOLE.print(
                f"[dim]hint: {gpu_type} on {partition} allows up to "
                f"{max_per_gpu} CPUs/GPU; {gpu_count} GPU(s) -> "
                f"max {max_per_gpu * gpu_count} total CPUs[/dim]"
            )
    cpus_per_task = _ask_int("cpus-per-task", default=cpus_default)

    mem = questionary.text(
        "Memory (e.g. 4G, 16G, 500M)", default=profile.get("mem", "4G")
    ).unsafe_ask()

    walltime = questionary.text(
        "Walltime ([D-]HH:MM:SS)",
        default=profile.get("walltime", "1:00:00"),
    ).unsafe_ask()

    constraint = questionary.select(
        "Architecture constraint",
        choices=["(none)", "epyc3", "epyc4"],
        default=profile.get("constraint") or "(none)",
    ).unsafe_ask()
    constraint = None if constraint == "(none)" else constraint

    chdir = questionary.text(
        "chdir (working directory; blank for default)",
        default=profile.get("chdir") or str(Path.cwd()),
    ).unsafe_ask()
    chdir = chdir.strip() or None

    source_files_raw = questionary.text(
        "source_files (space-separated paths to source before running; blank for none)",
        default=" ".join(profile.get("source_files", []) or []),
    ).unsafe_ask()
    source_files = [s for s in source_files_raw.split() if s]

    inline_script_args = questionary.text(
        "inline script args (CLI args appended after the python invocation)",
        default="",
    ).unsafe_ask()

    python_cmd = _ask_python_cmd(profile)

    modules_default: list[str] = list(profile.get("modules", []) or [])
    if "apptainer" in python_cmd and gres:
        wanted = "rocm" if partition == "gpu_rocm" else "cuda"
        if not any(m.split("/")[0] == wanted for m in modules_default):
            modules_default.append(wanted)

    modules_raw = questionary.text(
        "Modules to load (space-separated; e.g. 'cuda rocm/VERSION')",
        default=" ".join(modules_default),
    ).unsafe_ask()
    modules = [m for m in modules_raw.split() if m]

    env_vars = _ask_env_vars(profile, python_cmd)

    return BunyaSlurmJobSpec(
        job_name=job_name,
        account=account,
        partition=partition,
        qos=qos,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        cpus_per_task=cpus_per_task,
        mem=mem,
        walltime=walltime,
        gres=gres,
        constraint=constraint,
        chdir=chdir,
        source_files=source_files,
        modules=modules,
        env_vars=env_vars,
        inline_script_path=str(file_path),
        inline_script_args=inline_script_args,
        python_cmd=python_cmd,
    )


def _ask_env_vars(profile: dict[str, Any], python_cmd: str) -> dict[str, str]:
    """Free-text KEY=VALUE prompt for env vars to export in the SLURM job.

    Bare keys (no `=`) forward the value from the *current* shell — handy
    for `MLFLOW_TRACKING_URI` etc. that you've already exported.

    When python_cmd is apptainer-based, the script generator also emits
    APPTAINERENV_<KEY> for each var so they cross into the SIF.
    """
    last = " ".join(f"{k}={v}" for k, v in profile.get("env_vars", {}).items())
    raw = questionary.text(
        "Env vars (KEY=VALUE or bare KEY to forward from current shell; "
        "space-separated; blank for none)",
        default=last,
    ).unsafe_ask()

    env: dict[str, str] = {}
    for tok in raw.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            env[k] = v
        elif tok in os.environ:
            env[tok] = os.environ[tok]
        else:
            CONSOLE.print(f"[yellow]skip {tok!r}: not set in current shell[/yellow]")

    if "apptainer" in python_cmd and env:
        CONSOLE.print(
            f"[dim]hint: apptainer detected — these {len(env)} var(s) will "
            "also be mirrored as APPTAINERENV_* so they reach inside the "
            "container.[/dim]"
        )

    return env


def _ask_python_cmd(profile: dict[str, Any]) -> str:
    """Build the `python_cmd` string from a small choice menu.
    This is the python environment to run the script with.
    """
    last = profile.get("python_cmd", "python")
    choice = questionary.select(
        "How should python be invoked?",
        choices=[
            "system python (just `python`)",
            "specific venv (path to venv root)",
            "apptainer SIF (container)",
            "custom (paste exact command prefix)",
        ],
        default=(
            "apptainer SIF (container)"
            if "apptainer" in last
            else "specific venv (path to venv root)"
            if "/bin/python" in last
            else "system python (just `python`)"
        ),
    ).unsafe_ask()

    if choice.startswith("system"):
        return "python"

    if choice.startswith("specific venv"):
        default_venv = (
            last.removesuffix("/bin/python") if "/bin/python" in last else ""
        )
        venv_root = questionary.text(
            "Path to venv root (the dir containing bin/python)",
            default=default_venv,
            validate=lambda p: bool(p)
            and Path(p, "bin", "python").exists()
            or "no bin/python under that path",
        ).unsafe_ask()
        return str(Path(venv_root).resolve() / "bin" / "python")

    if choice.startswith("apptainer"):
        sif = questionary.text(
            "Path to .sif image",
            validate=lambda p: bool(p)
            and Path(p).exists()
            or "file not found",
        ).unsafe_ask()
        gpu = questionary.select(
            "GPU passthrough flag",
            choices=["(none)", "--nv  (NVIDIA / CUDA)", "--rocm (AMD / ROCm)"],
            default="(none)",
        ).unsafe_ask()
        gpu_flag = "" if gpu.startswith("(none)") else gpu.split()[0]
        binds_raw = questionary.text(
            "Bind mounts (space-separated, src[:dst]; blank = sane defaults)",
            default="/scratch /QRISdata",
        ).unsafe_ask()
        bind_flags = " ".join(f"-B {b}" for b in binds_raw.split() if b)
        parts = ["apptainer exec"]
        if gpu_flag:
            parts.append(gpu_flag)
        if bind_flags:
            parts.append(bind_flags)
        parts.extend([sif, "python"])
        return " ".join(parts)

    # custom
    return questionary.text(
        "Custom python command (everything before the script path)",
        default=last,
    ).unsafe_ask()


def _print_summary(spec: BunyaSlurmJobSpec) -> None:
    table = Table(title="SLURM job spec", show_header=False, box=None)
    table.add_column("field", style="cyan")
    table.add_column("value", style="white")
    fields = [
        "job_name",
        "account",
        "partition",
        "qos",
        "nodes",
        "ntasks_per_node",
        "cpus_per_task",
        "mem",
        "walltime",
        "gres",
        "constraint",
        "chdir",
        "source_files",
        "modules",
        "env_vars",
        "inline_script_path",
        "inline_script_args",
        "python_cmd",
    ]
    for f in fields:
        v = getattr(spec, f)
        if v is None or v == [] or v == "" or v == {}:
            continue
        if f == "env_vars":
            v = ", ".join(f"{k}={_mask(k, val)}" for k, val in v.items())
        table.add_row(f, str(v))
    CONSOLE.print(table)


def _mask(key: str, value: str) -> str:
    """Mask values whose key looks secret-shaped, for summary printing."""
    secret_markers = ("PASSWORD", "TOKEN", "KEY", "SECRET", "API_KEY")
    if any(m in key.upper() for m in secret_markers):
        return f"{value[:4]}…(masked, len={len(value)})"
    return value


def _validate_file(file_path: Path) -> int | None:
    """Shared file checks; returns an exit code on hard failure, else None."""
    if not file_path.exists():
        CONSOLE.print(f"[red]file not found:[/red] {file_path}")
        return 1
    if file_path.suffix != ".py":
        CONSOLE.print(
            f"[yellow]warning:[/yellow] {file_path} is not a .py — "
            "inline_script_path expects Python."
        )
    return None


def _build_spec_interactive(file_path: Path) -> BunyaSlurmJobSpec | int:
    """Run the interactive prompt loop; returns a spec or an exit code."""
    while True:
        try:
            return _build_spec(file_path)
        except ValidationError as e:
            CONSOLE.print(f"[red]validation failed:[/red]\n{e}")
            if not questionary.confirm("Try again?", default=True).unsafe_ask():
                return 2
        except KeyboardInterrupt:
            CONSOLE.print("\n[yellow]aborted[/yellow]")
            return 130


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive sbatch wrapper for Bunya."
    )
    parser.add_argument("file", type=Path, help="Python file to run via SLURM.")
    parser.add_argument(
        "--show-script", action="store_true", help="Print the rendered script."
    )
    args = parser.parse_args()

    rc = _validate_file(args.file)
    if rc is not None:
        return rc

    spec = _build_spec_interactive(args.file)
    if isinstance(spec, int):
        return spec

    _print_summary(spec)
    script = spec.to_script_content()
    if args.show_script:
        CONSOLE.print(Syntax(script, "bash", line_numbers=False))

    if not questionary.confirm("Submit via local sbatch?", default=True).unsafe_ask():
        CONSOLE.print("[yellow]not submitted. script:[/yellow]")
        CONSOLE.print(Syntax(script, "bash", line_numbers=False))
        return 0

    _save_profile(spec)

    result = subprocess.run(
        ["sbatch"], input=script, text=True, capture_output=True
    )
    if result.returncode != 0:
        CONSOLE.print(f"[red]sbatch failed:[/red]\n{result.stderr}")
        return result.returncode

    CONSOLE.print(f"[green]{result.stdout.strip()}[/green]")
    return 0


def create_main() -> int:
    """Like `run-slurm`, but writes the rendered sbatch script to a .sh file
    instead of submitting it. The script can later be submitted with `sbatch`.
    """
    parser = argparse.ArgumentParser(
        description="Interactively build a Bunya sbatch script and write it to a .sh file."
    )
    parser.add_argument("file", type=Path, help="Python file to run via SLURM.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to write the .sh script (default: <job_name>.sh in the cwd).",
    )
    parser.add_argument(
        "--show-script", action="store_true", help="Print the rendered script."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    rc = _validate_file(args.file)
    if rc is not None:
        return rc

    spec = _build_spec_interactive(args.file)
    if isinstance(spec, int):
        return spec

    _print_summary(spec)
    script = spec.to_script_content()
    if args.show_script:
        CONSOLE.print(Syntax(script, "bash", line_numbers=False))

    out_path = args.output or Path.cwd() / f"{spec.job_name}.sh"
    if out_path.suffix != ".sh":
        out_path = out_path.with_suffix(".sh")

    if out_path.exists() and not args.force:
        overwrite = questionary.confirm(
            f"{out_path} exists — overwrite?", default=False
        ).unsafe_ask()
        if not overwrite:
            CONSOLE.print("[yellow]not written.[/yellow]")
            return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script)
    out_path.chmod(0o755)

    _save_profile(spec)

    CONSOLE.print(f"[green]wrote sbatch script to {out_path}[/green]")
    CONSOLE.print(f"[dim]submit it with:[/dim] sbatch {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())