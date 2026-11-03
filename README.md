# CCD: Certified Breakdown Radii for Leaderboards

This repository contains the code for computing certified breakdown radii of leaderboard orderings, together with confidence intervals for the radius and the experiments on public evaluation records.

For two models A and B scored on the same n items, let g_i = loss_A(i) - loss_B(i), with A the observed winner (mean of g below 0). The radius is

    rho* = -min_{beta > 0} log mean_i exp(beta * g_i),

the smallest Kullback-Leibler divergence from the empirical item distribution to a reweighting under which B is at least as good as A. The minimizing exponential tilt is the reweighting that reverses the ordering. An ordering is certified at level epsilon when n * rho* >= log(1/epsilon).

## Layout

- `src/ccb.py`: radius, witness tilt and rank certificate for general loss differences.
- `src/ccd_fast.py`, `src/ccd_infer.py`: safeguarded Newton solver, plug-in variance, variance-stabilized and bootstrap intervals for the radius.
- `src/btlib.py`: Bradley-Terry fitting and the influence-function contrast used for preference leaderboards.
- `src/hfio.py`, `src/loader.py`, `src/loader2.py`: download and parsing of per-sample records from the Hugging Face Hub.
- `src/v2_fetch.py`, `src/v2_build.py`, `src/v2_ccb.py`: Open LLM Leaderboard v2 records, loss matrices and radii.
- `src/legacy_ccb.py`, `src/legacy_rank.py`: pairwise and rank certificates on the retired Open LLM Leaderboard archive.
- `src/arena_fetch.py`, `src/arena_ccb.py`: LMArena human preference votes and Bradley-Terry radii.
- `src/ccd_sim.py`: simulations for coverage, bootstrap behavior, the local limit, variance estimation and rank inference.
- `src/ccd_real.py`: intervals for the radii on the benchmark and arena records.
- `src/ccd_shift.py`: held-out subject shift study on MMLU.
- `src/ccd_figdata.py`: summary data for plots.
- `experiments/`: data acquisition and analysis scripts for the retired archive (`run16.py` builds `cache_lc.npz`).

## Requirements

Python 3.10 or later with the packages in `requirements.txt`:

    pip install -r requirements.txt

## Usage

Scripts in `src/` read cached loss matrices from the directory given by the environment variable `CCD_ROOT` (default: the current directory). Scripts in `experiments/` are run from inside that directory. Examples:

    cd src
    export CCD_ROOT=/path/to/data
    python ccd_sim.py --exp cover --out out_sim
    python ccd_real.py --part bench --B 4000 --out out_real
    python ccd_real.py --part arena --B 2000 --out out_real
    python ccd_shift.py --nsplit 400 --out shift.json

`ccd_sim.py` accepts `--task` and `--ntask` to split the designs across jobs, and the multiprocessing scripts use all available cores (or `SLURM_CPUS_PER_TASK` when set).

## License

MIT, see `LICENSE`.
