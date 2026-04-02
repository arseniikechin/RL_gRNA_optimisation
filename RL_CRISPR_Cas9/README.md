# A reinforcement learning agent discovers thermodynamically grounded mutation rules for CRISPR guide RNA optimisation.

The project trains a PPO agent to edit 20-nt guides and improve a composite objective:
- CRISPRspec surrogate (off-target specificity),
- GC-content preference,
- homopolymer penalty.

It also includes inference and evaluation scripts for reproducible benchmarking.

## Repository Layout

- `RL/train_grna_rl.py` - main training script (PPO).
- `RL/run_optimize_grna.py` - apply a trained model to CSV guides.
- `RL/analyze_policy.py` - policy analysis.
- `RL/metrics/compute_eval_100_doench_offtarget_cfd_crisprbert.py` - benchmark metrics (Doench/off-target/CFD/CRISPR-BERT).
- `RL/grna_gym_env.py` - Gym environment and reward logic.
- `data/` - current input/reference datasets used by scripts.
- `energy/` - CRISPRspec energy pipeline resources.

## Environment Setup

Use Python 3.10+ (3.10/3.11 recommended).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install --upgrade pip
pip install numpy pandas biopython regex gymnasium stable-baselines3[extra] torch
pip install rpy2 tensorflow keras-bert
```

Notes:
- `rpy2` requires a working local R installation.
- GPU is optional; scripts can run on CPU (`--force-cpu` or `--device cpu`).

## Data Currently Kept in `data/`

## Quick Start

### 1) Train a new policy

Run this script from RL folder.

```bash
python train_grna_rl.py ^
  --sequences ../data/train_400.txt ^
  --steps 200000 ^
  --seed-len 8 ^
  --max-episode-steps 20 ^
  --n-envs 4 ^
  --use-crisprspec ^
  --reference-fasta ../data/chr_3_7_20_21.fna
```

Model output (default): `RL/models/grna_ppo.zip`


### 2) Optimize guides from CSV

Input CSV must contain column `sgRNA` (20-nt guides).
Run this script from RL folder.

```bash
python run_optimize_grna.py ^
  --model models/grna_ppo.zip ^
  --input ../data/run_normal_top50_sgRNA_with_crisprbert.csv ^
  --output ../data/optimized_sgrna.csv ^
  --reference-fasta ../data/genes_with_flankers.fna
```

### 3) Analyze trained policy (statistics-only outputs)

Run this script from RL folder.

```bash
python analyze_policy.py ^
  --model models/grna_ppo.zip ^
  --sequences ../data/eval_100.txt ^
  --genome ../data/chr_3_7_20_21.fna ^
  --output analysis_results
```

Outputs:
- `analysis_results/policy_report.txt`
- `analysis_results/policy_summary.json`

### 4) Compute benchmark metrics

Run this script from metrics folder.

```bash
python compute_eval_100_doench_offtarget_cfd_crisprbert.py ^
  --input ../data/eval_100_optimized.csv ^
  --output ../data/eval_100_optimized_metrics.csv ^
  --genome ../data/genes_with_flankers.fna
```

## Reproducibility Checklist

- Keep `seed-len`, `max-mismatches`, and reward weights consistent between training and inference.
- Use the same reference FASTA for train/eval whenever possible.
- Save run metadata (`logs/`, `training_summary.json`, CLI command lines) for each experiment.
- Report exact model checkpoint (`.zip`) used for inference and policy analysis.

## Common Issues

- `stable-baselines3 not installed`:
  - install: `pip install stable-baselines3[extra]`
- CRISPRspec scoring disabled:
  - check FASTA path and pass `--reference-fasta data/genes_with_flankers.fna`
- CSV load errors in optimization:
  - ensure `sgRNA` column exists and sequences are valid 20-mers (`A/C/G/T`).

