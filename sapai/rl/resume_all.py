"""
Resume every training run and the benchmark watcher after they stopped, for
example after the machine restarted.

    python -m sapai.rl.resume_all            # resume everything that stopped
    python -m sapai.rl.resume_all --dry-run  # only show the commands

For each series of runs (runs/mlp1, runs/mlp2, ... is the series "mlp"), the
newest run folder with an args.json and a latest_model.zip is continued in the
next folder (runs/mlp3) with the same settings, from its latest model and with
its saved self-play pool. Runs that are still training are left alone. Then
the benchmark watcher is restarted with the same settings, watching the new
run folders.

Jobs run in the background; their output goes to runs/<folder>.log and
runs/benchmark.log.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

from sapai.rl.benchmark import watch_args_path


def series_of(name):
    """("mlp", 6) for "mlp6", or None for folders that are not numbered runs"""
    match = re.fullmatch(r"(.*?)(\d+)", name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def latest_runs(runs_dir):
    """{series: (number, folder)} of the newest resumable run of each series"""
    latest = {}
    for name in os.listdir(runs_dir):
        folder = os.path.join(runs_dir, name)
        parsed = series_of(name)
        if not parsed or not os.path.isdir(folder):
            continue
        if not os.path.exists(os.path.join(folder, "args.json")):
            continue
        if not os.path.exists(os.path.join(folder, "latest_model.zip")):
            continue
        series, number = parsed
        if series not in latest or number > latest[series][0]:
            latest[series] = (number, folder)
    return latest


def running_commands():
    """Command lines of the running processes"""
    try:
        out = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return out.splitlines()


def is_running(commands, *needles):
    return any(all(n in line for n in needles) for line in commands)


def train_command(args, folder, new_folder):
    """Training command continuing folder in new_folder with the same settings"""
    args = dict(args)
    resume = os.path.join(folder, "model_resume.zip")
    args["resume"] = resume
    pool = os.path.join(folder, "selfplay_teams.pkl")
    args["selfplay_teams"] = pool if os.path.exists(pool) else None
    args["out"] = new_folder
    command = [sys.executable, "-m", "sapai.rl.train"]
    for key, value in args.items():
        if value is None:
            continue
        command.append(f"--{key.replace('_', '-')}={value}")
    return command, resume


def benchmark_command(args, run_dirs):
    args = dict(args)
    command = [sys.executable, "-m", "sapai.rl.benchmark", "--watch", *run_dirs]
    command += ["--pool", *args["pool"]]
    for key in ["games", "turns", "seed", "every_minutes", "out_dir", "csv"]:
        command.append(f"--{key.replace('_', '-')}={args[key]}")
    return command


def launch(command, log_path, dry_run):
    print(("would run: " if dry_run else "running: ") + " ".join(command))
    if dry_run:
        return
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(log_path, "a") as log:
        subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--benchmark-csv", default="runs/benchmark.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    commands = running_commands()
    watched = []
    for series, (number, folder) in sorted(latest_runs(args.runs_dir).items()):
        if is_running(commands, "sapai.rl.train", folder):
            print(f"{folder} is still training")
            watched.append(folder)
            continue
        with open(os.path.join(folder, "args.json")) as f:
            run_args = json.load(f)
        new_folder = os.path.join(args.runs_dir, f"{series}{number + 1}")
        command, resume = train_command(run_args, folder, new_folder)
        if not args.dry_run:
            shutil.copy(os.path.join(folder, "latest_model.zip"), resume)
        launch(command, new_folder + ".log", args.dry_run)
        watched.append(new_folder)

    watch_path = watch_args_path(args.benchmark_csv)
    if not os.path.exists(watch_path):
        print("No benchmark watcher settings found, not starting one")
    elif is_running(commands, "sapai.rl.benchmark", "--watch"):
        print("The benchmark watcher is still running")
    else:
        with open(watch_path) as f:
            watch_args = json.load(f)
        ### Watch the newest folder of every series that was watched before
        old = [
            series_of(os.path.basename(os.path.normpath(x)))
            for x in watch_args["watch"]
        ]
        old_series = {x[0] for x in old if x}
        run_dirs = [
            x for x in watched if series_of(os.path.basename(x))[0] in old_series
        ]
        log = os.path.join(os.path.dirname(args.benchmark_csv) or ".", "benchmark.log")
        launch(benchmark_command(watch_args, run_dirs), log, args.dry_run)


if __name__ == "__main__":
    main()
