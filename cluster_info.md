# ALICE HPC Cluster Reference for LLM Agents

This document provides everything an LLM agent needs to adapt a locally-running project for the ALICE HPC cluster at Leiden University. Use this as context when helping a user deploy their code on ALICE.

## Cluster Overview

- **Name**: ALICE (Academic Leiden Interdisciplinary Cluster Environment)
- **Scheduler**: SLURM
- **Module system**: Lmod (hierarchical — requires `ALICE/default` loaded first)
- **SSH gateway**: `ssh-gw.alice.universiteitleiden.nl`
- **Login nodes**: `login.alice.universiteitleiden.nl` (e.g., `nodelogin03`)

## Filesystem

- **Home directory**: `/zfsstore/user/<username>` (mounted at `$HOME`)
- **Storage backend**: ZFS on `datastore23.data.leidenuniv.nl`
- **Quota**: 2 TB per user
- **Shared across all nodes**: Yes — login nodes and compute nodes see the same filesystem. No need to copy files between nodes.
- **Check usage**: `df -h $HOME`

### Where to Store What

| Content | Location | Notes |
|---------|----------|-------|
| Code/project files | `~/GitHub/<project>/` or `~/projects/<project>/` | Any path under `$HOME` works |
| Virtual environments | `~/.venvs/<project_name>/` | Keeps venvs separate from code |
| Results & checkpoints | `~/<project>/results/` | Written by jobs, stays on shared storage |
| SLURM logs | `~/<project>/slurm_logs/` | Created by sbatch `--output`/`--error` |

**Important**: There is no separate scratch filesystem. Everything lives under `$HOME` on ZFS.

## Available Partitions

| Partition | Type | Time Limit | Nodes | Hardware | Notes |
|-----------|------|-----------|-------|----------|-------|
| `cpu-short` | CPU | 4 hours | 48 | Mixed | Default CPU queue, all users |
| `cpu-zen4` | CPU | 7 days | 12 | AMD Zen4 | Newer CPU nodes |
| `cpu-skylake` | CPU | 7 days | 19 | Intel Skylake | Older CPU nodes |
| `gpu-short` | GPU | 4 hours | 30 | Mixed GPUs | Quick GPU jobs, testing |
| `gpu-a100-80g` | GPU | 7 days | 6 | NVIDIA A100 80GB PCIe | Best for deep learning |
| `gpu-mig-40g` | GPU | 7 days | 7 | A100 MIG 40GB slices | Smaller GPU jobs |
| `gpu-l4-24g` | GPU | 7 days | 7 | NVIDIA L4 24GB | Inference, light training |
| `gpu-2080ti-11g` | GPU | 7 days | 10 | NVIDIA 2080 Ti 11GB | Legacy, small models |
| `mem` | CPU | 14 days | 2 | Up to 2TB RAM | Large memory jobs |
| `testing` | Mixed | 30 min | 3 | Mixed | Quick tests only |
| `interactive` | Mixed | 8 hours | 4 | Mixed | Interactive sessions |

### Partition Selection Guide

| Workload | Recommended Partition | Why |
|----------|----------------------|-----|
| Deep learning training (large models, long runs) | `gpu-a100-80g` | 80GB VRAM, 7-day limit |
| Deep learning training (small models) | `gpu-l4-24g` or `gpu-2080ti-11g` | Sufficient VRAM, more availability |
| Quick GPU test (<4 hrs) | `gpu-short` | Faster queue, any GPU |
| CPU-only preprocessing, analysis, plotting | `cpu-short` | 4-hour limit covers most tasks |
| Long CPU jobs (>4 hrs) | `cpu-zen4` or `cpu-skylake` | 7-day limit |
| Large memory requirements (>240GB) | `mem` | Up to 2TB RAM |

## Module System

**Critical**: ALICE uses hierarchical Lmod. You must load `ALICE/default` before any other module.

```bash
module purge
module load ALICE/default
module load <module_name>
```

### Available Key Modules (as of March 2026)

**Python versions**:
- `Python/3.10.4-GCCcore-11.3.0`
- `Python/3.10.8-GCCcore-12.2.0`
- `Python/3.11.3-GCCcore-12.3.0`
- `Python/3.11.5-GCCcore-13.2.0`
- `Python/3.12.3-GCCcore-13.3.0`
- `Python/3.13.1-GCCcore-14.2.0`
- `Python/3.13.5-GCCcore-14.3.0`

**CUDA versions**:
- `CUDA/11.7.0`, `CUDA/11.8.0`
- `CUDA/12.1.1`, `CUDA/12.3.0`, `CUDA/12.3.2`, `CUDA/12.4.0`

**Driver**: NVIDIA 580.82.07 (supports up to CUDA 13.0)

To discover other modules: `module spider <name>`.

### Recommended Module Combinations

| Use Case | Python | CUDA | PyTorch Install |
|----------|--------|------|-----------------|
| Latest stable | `Python/3.11.3-GCCcore-12.3.0` | `CUDA/12.4.0` | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Maximum compat | `Python/3.10.8-GCCcore-12.2.0` | `CUDA/11.8.0` | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

## Account System

- GPU partitions require `--account=<PROJECT>` in job submissions.
- Users discover their account with:
  ```bash
  sacctmgr show associations user=$USER format=account,partition
  ```
- The account name is typically a department/group abbreviation (e.g., `liacs`, `strw`, `lion`).
- If the partition column is blank, the account has access to all partitions.

## SLURM Job Submission

### Required SBATCH Directives

Every sbatch script must include:

```bash
#!/bin/bash
#SBATCH --job-name=<name>          # Short descriptive name
#SBATCH --partition=<partition>     # See partition table above
#SBATCH --time=HH:MM:SS            # REQUIRED — jobs without --time are rejected
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<N>
#SBATCH --mem=<N>G
#SBATCH --output=slurm_logs/<name>_%j.out
#SBATCH --error=slurm_logs/<name>_%j.err
```

For GPU jobs, add:
```bash
#SBATCH --gres=gpu:<N>              # Number of GPUs (no type prefix needed on typed partitions)
```

**Important**: The `--account` flag is passed via `sbatch --account=<PROJECT>` at submission time, not hardcoded in the script. This keeps scripts portable.

### Array Jobs

For embarrassingly parallel workloads (e.g., training multiple models):

```bash
#SBATCH --array=0-11%4    # 12 tasks (0-11), max 4 running concurrently
```

Access the task index in the script via `$SLURM_ARRAY_TASK_ID`.

### Job Dependencies

Chain jobs so they run in order:

```bash
JOB1=$(sbatch --parsable job1.sbatch)
JOB2=$(sbatch --dependency=afterok:$JOB1 --parsable job2.sbatch)
```

Dependency types:
- `afterok:<id>` — run only if job succeeded
- `afterany:<id>` — run regardless of success/failure
- `afterok:<array_id>` — waits for ALL array tasks to succeed

### Common SLURM Commands

```bash
squeue -u $USER                    # View your jobs
squeue -u $USER --start            # Estimated start times for pending jobs
sacct -j JOBID --format=JobID,JobName,State,Elapsed,MaxRSS  # Job details
scancel JOBID                      # Cancel a job
scancel -u $USER                   # Cancel all your jobs
sinfo -s                           # Partition overview
scontrol show job JOBID            # Full job info (including why it's pending)
```

### Job States

| State | Code | Meaning |
|-------|------|---------|
| Pending | PD | Waiting for resources or dependency |
| Running | R | Executing |
| Completing | CG | Finishing up |
| Completed | CD | Finished successfully |
| Failed | F | Exited with error |
| Timeout | TO | Exceeded time limit |
| Cancelled | CA | Cancelled by user or admin |

## File Transfer

### Upload (local to ALICE)

```bash
rsync -avz --exclude='.venv/' --exclude='__pycache__/' --exclude='results/' \
    /path/to/local/project/ \
    <username>@ssh-gw.alice.universiteitleiden.nl:~/project_name/
```

### Download (ALICE to local)

```bash
# Everything
rsync -avz <username>@ssh-gw.alice.universiteitleiden.nl:~/project_name/results/ \
    /path/to/local/project/results/

# Exclude large files (e.g., model checkpoints)
rsync -avz --exclude='checkpoints/' \
    <username>@ssh-gw.alice.universiteitleiden.nl:~/project_name/results/ \
    /path/to/local/project/results/
```

### Troubleshooting File Transfer

`rsync` through the SSH gateway can fail with "unexpected end of file". Fallbacks:

```bash
# Option 1: Use scp instead
scp /path/to/file <username>@ssh-gw.alice.universiteitleiden.nl:~/remote/path/file

# Option 2: Edit directly on the cluster
ssh <username>@ssh-gw.alice.universiteitleiden.nl
nano ~/remote/path/file
```

### SSH Config (recommended)

Users should add to `~/.ssh/config`:

```
Host alice-gw
    HostName ssh-gw.alice.universiteitleiden.nl
    User <username>

Host alice
    HostName login.alice.universiteitleiden.nl
    User <username>
    ProxyJump alice-gw
```

Then: `ssh alice`, `rsync -avz alice:~/project/ ./project/`

**Note**: The user has this SSH config set up and uses the `alice` alias for all commands (e.g., `rsync -av file alice:~/QNetGame/file`).

## Managing Running Jobs

### Extending Time Limits

If a running job needs more time than originally requested:

```bash
scontrol update JobId=<JOBID> TimeLimit=20:00:00
```

This only works if the new limit doesn't exceed the partition maximum.

### Updating Config Mid-Pipeline

If you update a config file while an array job is running:
- **Already-running tasks**: unaffected — they loaded the config at startup
- **Pending tasks**: will pick up the new config when they start

This means you can safely change parameters (e.g., reduce ensemble size) and only future tasks will see the change.

### Cancelling and Resubmitting

```bash
scancel -u $USER                    # Cancel all your jobs
scancel <JOBID>                     # Cancel a specific job
scancel <ARRAY_JOBID>_<TASK_ID>     # Cancel a specific array task
```

After cancelling, clean up partial results before resubmitting to avoid stale data mixing with fresh results.

### Time Limit Estimation

**Critical**: Set `--time` with a buffer above expected runtime. If a job is killed by the time limit:
- Any results not yet written to disk are **lost**
- Programs that save only at completion (not incrementally) lose all progress

Rule of thumb: set `--time` to **1.5-2x** the expected runtime. Use early runs to calibrate:

```bash
# Check actual runtime of completed jobs
sacct -j <JOBID> --format=JobID,Elapsed,State
```

## Environment Setup

### Important: Do NOT set up on login nodes

Login nodes are shared and resource-limited. Running `pip install torch` or other heavy installations on login nodes is bad practice and may be killed or throttled. Always use a compute node.

### Option 1: Interactive session (recommended)

```bash
srun --partition=cpu-short --time=01:00:00 --cpus-per-task=4 --mem=16G \
    --account=<PROJECT> --pty bash
cd ~/path/to/project
bash slurm/setup_env.sh
exit
```

### Option 2: Submit as a batch job

Add `#SBATCH` headers to the setup script and submit with `sbatch`.

### Setup Script Template

```bash
#!/bin/bash
set -euo pipefail

module purge
module load ALICE/default
module load Python/<version>-GCCcore-<toolchain>
module load CUDA/<version>  # only if GPU needed

VENV_DIR="$HOME/.venvs/<project_name>"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu<XYZ>  # if needed
pip install -r requirements.txt
```

### Verifying GPU Access After Setup

Test from an interactive GPU session:

```bash
srun --partition=gpu-short --gres=gpu:1 --account=<PROJECT> --time=00:05:00 \
    bash -c 'source ~/.venvs/<project>/bin/activate && \
    python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
```

## Standard SBATCH Script Template

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --partition=<partition>
#SBATCH --time=<HH:MM:SS>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<N>
#SBATCH --mem=<N>G
#SBATCH --gres=gpu:<N>              # GPU jobs only
#SBATCH --output=slurm_logs/<name>_%j.out
#SBATCH --error=slurm_logs/<name>_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load ALICE/default
module load Python/<version>-GCCcore-<toolchain>
module load CUDA/<version>          # GPU jobs only
source "$HOME/.venvs/<project_name>/bin/activate"

echo "Job $SLURM_JOB_ID started at $(date)"

# --- Run your code here ---
python3 <script.py>

echo "Job completed at $(date)"
```

## Master Submission Script Template

```bash
#!/bin/bash
set -euo pipefail

PROJECT="$1"
mkdir -p slurm_logs

# Submit jobs with dependencies
JOB1=$(sbatch --account="$PROJECT" --parsable slurm/step1.sbatch)
echo "Step 1: Job $JOB1"

JOB2=$(sbatch --account="$PROJECT" --dependency=afterok:"$JOB1" --parsable slurm/step2.sbatch)
echo "Step 2: Job $JOB2 (depends on $JOB1)"

echo ""
echo "Monitor: squeue -u \$USER"
```

---

## Questions to Ask the User

When helping a user deploy their project on ALICE, gather the following information before generating scripts:

### Required

1. **What is your ALICE username?**
   _Needed for SSH/rsync commands. Example: `chalkiasc1`_

2. **What is your HPC account/project name?**
   _Run `sacctmgr show associations user=$USER format=account,partition` on ALICE to find it._

3. **What does your program do? Describe the pipeline steps.**
   _Identify which steps are independent (parallelizable) vs sequential (need dependencies)._

4. **Which steps need a GPU and which are CPU-only?**
   _Determines partition selection per step._

5. **What language/runtime does your project use?**
   _Python version, CUDA version, R, Julia, compiled C++, etc._

6. **How do you run the program locally?**
   _The exact commands — e.g., `python3 train.py --epochs 100`. This determines what goes in each sbatch script._

7. **What are the expected output files and where are they written?**
   _Verify output paths are relative (not absolute to a local machine)._

### Resource Estimation

8. **How long does each step take locally?**
   _Use as baseline to estimate cluster time. GPU steps are typically 10-20x faster on A100 vs laptop CPU._

9. **How much memory does each step need?**
   _If unknown, start with 16-32GB for CPU jobs, 64GB for GPU jobs._

10. **Are there any steps that can be parallelized (e.g., training multiple models)?**
    _Candidates for SLURM array jobs. Ask how many parallel tasks and what parameter varies._

11. **How much disk space will results consume?**
    _Model checkpoints can be large. Home quota is 2 TB (ZFS). Check usage with `df -h $HOME`._

### Environment

12. **Does the project have a `requirements.txt`, `pyproject.toml`, `environment.yml`, or similar?**
    _Needed to replicate the environment on the cluster._

13. **Does the project need any system libraries beyond Python/CUDA?**
    _E.g., FFmpeg, HDF5, MPI. Check `module spider <name>` on ALICE._

14. **Are there any hardcoded absolute paths in the code?**
    _These will break on the cluster. Must be converted to relative paths or config-driven._

### Optional

15. **Do you want email notifications when jobs start/finish/fail?**
    _Add `#SBATCH --mail-type=END,FAIL` and `#SBATCH --mail-user=<email>` if yes._

16. **Do you need to resume from checkpoints if a job times out?**
    _If yes, the code must support checkpoint/resume and scripts need restart logic._

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| `module load` fails with "cannot be loaded as requested" | Load `ALICE/default` first, then the target module |
| `--time` not specified | SLURM rejects jobs without a time limit — always set `--time` |
| Wrong partition name | Run `sinfo -s` to see actual partition names |
| Invalid account/partition combination | Run `sacctmgr show associations user=$USER format=account,partition` |
| `gpu-devel` partition doesn't exist | Use `gpu-short` (4hr) for quick GPU tests |
| Code uses absolute paths like `/Users/...` | Convert to relative paths or use config files |
| Output directory doesn't exist on cluster | Add `mkdir -p <output_dir>` in the sbatch script or the code |
| PyTorch doesn't see GPU | Ensure CUDA module is loaded and PyTorch was installed with matching CUDA version |
| Job stuck in `PD` with `(Dependency)` | Normal — waiting for a prior job. Check with `scontrol show job JOBID` |
| Job OOM killed | Increase `--mem` or reduce batch size |
| Array job index mismatch | `$SLURM_ARRAY_TASK_ID` is 0-indexed by default — match to your data |
| Large checkpoint files fill quota | Check `df -h $HOME` (2 TB quota on ZFS); delete checkpoints after analysis if not needed |
| Running `pip install` on login node | Use an interactive compute session: `srun --partition=cpu-short --time=01:00:00 --cpus-per-task=4 --mem=16G --account=<PROJECT> --pty bash` |
| Forgot `--pty bash` on interactive session | Without `--pty bash`, `srun` runs a single command, not an interactive shell |
| `rsync` fails through SSH gateway | Use `scp` instead, or SSH in and edit files directly with `nano` |
| `ModuleNotFoundError` for project packages | Add `export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"` before the python call in the sbatch script |
| Job killed at time limit, results lost | Programs that save only at completion lose all progress. Set `--time` to 1.5-2x expected runtime. Consider adding incremental checkpointing to long-running code |
| Changed config but running jobs use old values | Already-running tasks loaded config at startup. Only pending/future tasks see the new config |
| `scontrol update` to extend time rejected | New time limit cannot exceed the partition's max time limit |
| Sourcing module init scripts in sbatch | **Never source `/usr/share/modules/init/bash`, `/etc/bashrc`, or `/etc/profile.d/lmod.sh` in sbatch scripts.** SLURM already exports the `module` bash function and `MODULEPATH` from the login shell. Sourcing init scripts overwrites or clears them, breaking everything. |
| `module: command not found` in batch job | Do NOT try to re-initialize the module system. The function is exported automatically via `BASH_FUNC_module%%`. Just call `module purge` directly — no sourcing needed. |
| `MODULEPATH is undefined` after `module purge` | Caused by sourcing an init script that reset MODULEPATH before the purge. Fix: remove the source call entirely. |
| `testing` and `gpu-short` partitions have no modules | Confirmed: these partitions do not export the `module` bash function and lack `/usr/share/modules/init/bash`. Never use them for jobs that call `module`. |
| Module init differs by partition | `gpu-l4-24g`, `gpu-a100-80g`, and `cpu-zen4` all export `module` as a bash function with `MODULEPATH` set — call `module purge` directly, no sourcing needed. `gpu-short` and `testing` do NOT export it and lack init files; never use them for jobs that need modules. |
| Diagnosing module system on a compute node | Run: `srun --partition=<p> --account=<proj> --time=00:05:00 --gres=gpu:1 bash -c 'echo "MODULEPATH=${MODULEPATH:-UNSET}"; type module 2>&1; ls /etc/profile.d/ \| grep -iE "lmod\|module"'` |
| `source /etc/bashrc` causes `BASHRCSOURCED: unbound variable` | `/etc/bashrc` uses variables that are unset in batch job shells. Never source it. |
| GPU partition requires `--gres` in `srun` diagnostic commands | `srun` on `gpu-*` partitions requires `--gres=gpu:<count>` or the allocation fails immediately. |
