"""
Waits for unfinished experiments and automatically completes them.
"""

import time
import random
import ast
import subprocess
import random

import wandb

from Misc.config import config
from Misc.utils import transform_dict_to_args_list
from benchmark import main as benchmark
from collect_results import is_eval_done_and_get_sorted_runs

max_wait_period_in_seconds = 20


def get_git_commit_hash():
    """
    Gets the commit hash of this repository
    """
    try:
        # Get the current commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError:
        return None

def get_unfinished_IDs():
    api = wandb.Api()
    sweeps = api.project(config.project).sweeps()

    # Filter sweeps whose names start with "umi_"
    filtered_sweeps = [sweep for sweep in sweeps if sweep.name.startswith("umi_")]
    unfinished_sweeps = []
    
    for sweep in filtered_sweeps:
        id = sweep.id
        
        # Finished: check if evaluation is done
        if sweep.state.lower() == "finished":
            # Check if all runs in sweep are also finished
            do_skip_sweep = False
            for run in sweep.runs:
                if run.state.lower() == "running":
                    print(f"Found a not finished experiment ({id}), but sweep still has unfinished runs")
                    do_skip_sweep = True
                    break
                
                # Case: "preempting", "preempted"
                # -> enter sweep and see if we need to delete something
            
            if do_skip_sweep:
                continue
            
            finished, _ = is_eval_done_and_get_sorted_runs(api, sweep)
            if finished:
                continue

        # Check if the run is still going (so we can help)
        elif not(sweep.state.lower() in ["running", "pending"]):
            continue
    
        # We found an unfinished run, check if it uses the same git hash
        runs_in_sweep = sweep.runs
        
        if len(runs_in_sweep) > 0 and not config.disable_auto_git_safeguards:
            run_git_hash = runs_in_sweep[0].commit
            
            # Run is not completely initialized -> skip it for now
            if run_git_hash is None:
                continue
            
            local_git_hash = get_git_commit_hash()
            if run_git_hash == local_git_hash:
                unfinished_sweeps.append(id)
            else:
                print("Found an unfinished run but it uses a different commit:")
                print(f"\tRun commit hash: {run_git_hash}")
                print(f"\tLocal commit hash: {local_git_hash}")
            continue
        else:
            unfinished_sweeps.append(id)
            continue
        
    return unfinished_sweeps

def benchmark_ID(id):
    api = wandb.Api()
    sweep = api.sweep(config.project + "/" + id)
    params_dict = ast.literal_eval(sweep.config['description'])
    params_dict["--sweep"] = id
    params_dict["-project"] = config.project
    benchmark(transform_dict_to_args_list(params_dict))

def main():
    while True:
        print("Collecting unfinished experiments")
        unfinished_sweeps = get_unfinished_IDs()

        if len(unfinished_sweeps) > 0:
            for id in unfinished_sweeps:
                print(f"Found unfinished experiment: {id}")
                benchmark_ID(id)

                # In case we need to go back to a previous run, we stop looping this with a 50% chance
                if random.choice([True, False]):
                    break
        else:
            print("No unfinished experiments found")
        
        # Sleep for a somewhat random time to avoid syncing of different runs
        wait_time = max_wait_period_in_seconds * random.random()
        print(f"Waiting {wait_time:.2f} seconds")
        time.sleep(wait_time)


if __name__ == "__main__":
    main()