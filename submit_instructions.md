# Submitting a Training Job to ALICE

## 1. Upload changed files

```bash
rsync -av rl_stack/ alice:~/QNetGame/rl_stack/
rsync -av train-test/ alice:~/QNetGame/train-test/
rsync -av submit.sh alice:~/QNetGame/submit.sh
```

Or sync the whole project (respecting `.gitignore`):

```bash
rsync -av --exclude='.git' --filter=':- .gitignore' . alice:~/QNetGame/
```

## 2. Update `submit.sh` parameters

Edit `submit.sh` before uploading (or on the cluster) to set the run config.
Key fields to check each time:

| Field | What to set |
|-------|------------|
| `--run_id` | Unique name for this run (e.g. `cluster_002`) |
| `--episodes` | Training length (default now 50000) |
| `--p_gen` | Generation probability (default now 0.40) |
| `--cutoff` | Link lifetime (default now 6) |
| `--partition` | `gpu-short` (quick tests) or `gpu-a100-80g` (long runs) |
| `--time` | Wall time limit matching your partition |

## 3. Submit the job

```bash
ssh alice
cd ~/QNetGame
sbatch --account=liacs submit.sh
```

## 4. Monitor

```bash
squeue -u $USER          # check job status
tail -f slurm_logs/train_<JOBID>.out   # live output
```

## 5. Download results

```bash
rsync -av alice:~/QNetGame/checkpoints/<run_id>/ checkpoints/<run_id>/
```

## 6. bashrc

```bash
cat >> ~/.bashrc << 'EOF'

# --- Aliases ---
alias ll='lsd -la'
alias lt='lsd --tree --depth 2'
alias q='squeue -u $USER'
alias qa='sacct -u $USER --format=JobID,JobName,Partition,State,ExitCode,Elapsed -X'
alias myquota='quota -s'

# --- SLURM shortcuts ---
alias sub='sbatch --account=liacs'
alias scanc='scancel -u $USER'
alias slog='tail -f slurm_logs/train_*.out'

# --- Navigation ---
alias proj='cd ~/QNetGame'
alias ..='cd ..'
alias ...='cd ../..'

# --- History ---
HISTSIZE=10000
HISTFILESIZE=20000
HISTCONTROL=ignoredups:erasedups
shopt -s histappend

# --- Misc ---
export EDITOR=vim
alias grep='grep --color=auto'

EOF

source ~/.bashrc
```
