# bunya-slurm-tui

Interactive TUI for submitting python scripts as SLURM jobs.

Constrains parameters based on [UQ-RCC Bunya](https://github.com/UQ-RCC/hpc-docs).

Steps through SLURM job parameters with questionary prompts, validates the
combination against documented Bunya limits (GPU/partition/QoS/CPU compatibility)
via pydantic, and pipes the rendered script to local `sbatch`.

Currently designed to run within a compute node, where you can already invoke `sbatch` from
the shell. In future, potentially call using REST API / slurmrestd.

## Install

```bash
uv tool install --from /path/to/bunya_slurm_tui run-slurm
# or, from a clone:
cd bunya_slurm_tui && uv sync && uv run run-slurm SCRIPT.py
```

## Use

```bash
run-slurm scripts/2_train.py
```

The TUI walks you through:

1. Job name, account
2. **GPU?** -> if yes, GPU type -> partition (auto-resolved if only one valid) -> GPU count
3. QoS (filtered by partition)
4. Resources (nodes, ntasks, cpus-per-task, memory, walltime)
5. Constraint, chdir, source files, inline-script args
6. **Python invocation** — system `python`, specific venv, or apptainer SIF
   (with `--nv`/`--rocm` and bind-mount prompts)
7. Summary table -> confirm -> `sbatch`

Defaults are pulled from `~/.config/run-slurm/last.json` (your previous
successful submission), so re-runs only need overrides.

## Constraints encoded

The pydantic model `BunyaSlurmJobSpec` enforces:

- account string starts with `a_`
- `mem` is `\d+[KMGT]?`, walltime is `[D-]HH:MM:SS`
- exactly one of `script`/`command`/`inline_script_path` set
- gres ↔ partition compatibility (per UQ-RCC's GPU-partition matrix)
- partition ↔ QoS compatibility
- per-GPU CPU limits (e.g. `a100` on `gpu_cuda` allows ≤ 86 CPUs/GPU)

See: <https://github.com/UQ-RCC/hpc-docs> Bunya User Guide.

## Library use

The validator and script generator are usable without the TUI:

```python
from bunya_slurm_tui import BunyaSlurmJobSpec

spec = BunyaSlurmJobSpec(
    job_name="train",
    account="a_my_alloc",
    partition="gpu_cuda",
    qos="gpu",
    gres="gpu:a100:1",
    cpus_per_task=16,
    mem="64G",
    walltime="08:00:00",
    inline_script_path="scripts/2_train.py",
)
print(spec.to_script_content())
```