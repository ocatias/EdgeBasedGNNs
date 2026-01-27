"""
Collect results of finished experiments from W&B
"""

import ast
import json
import os

import wandb

from utils import *

def is_eval_done_and_get_sorted_runs(api, sweep):
    """
    Returns whether we have enough eval runs and returns all relevant runs (also deals with the case of too many eval runs)
    """
    id = sweep.id
    eval_tag = f"{id}--EVAL"

    runs = api.runs(f"{sweep.entity}/{wandb_project}")

    # Run needs to have the eval_tag
    runs_with_eval_tag = [run for run in runs if eval_tag in run.tags]
    
    # Only include finished runs
    finished_runs = [run for run in runs_with_eval_tag if (run.state.lower() == "finished" or run.state.lower() == "running")]
    repeats = ast.literal_eval(sweep.config['description'])["--repeats"]
    
    # Not enough finished runs -> experiment not done
    if len(finished_runs) < repeats:
        return False, finished_runs
    
    # If we have too many eval runs only use as many as necessary
    if len(finished_runs) > repeats:
        sorted_runs = sorted(finished_runs, key=lambda run: run.created_at)
        finished_runs = sorted_runs[:repeats]
    assert len(finished_runs) == repeats
    
    return True, finished_runs

def get_result_of_experiment(eval_runs):
    results = []
    for run in eval_runs:
        run_results = dict(run.config)
        for key in run.summary.keys():
            if key == "_wandb":
                continue
            run_results[key] = run.summary[key]
            
        results.append(run_results)
        
    return results

def main():
    api = wandb.Api()
    sweeps = api.project(wandb_project).sweeps()
    filtered_sweeps = [sweep for sweep in sweeps if sweep.name.startswith("umi_")]
    results = []
    
    for sweep in filtered_sweeps:
        result = {"eval": None, "tune": None}
        finished, eval_runs = is_eval_done_and_get_sorted_runs(api, sweep)
        
        # Collect tuning results
        tuning_results = []
        for run in sweep.runs:
            if run.state != "finished":
                continue
            
            run_results = dict(run.config)
            for key in run.summary.keys():
                if key == "_wandb":
                    continue
                run_results[key] = run.summary[key]
                
            tuning_results.append(run_results)
        result["tune"] = tuning_results
        
        # Collect evaluation result
        if finished:
           result["eval"] = get_result_of_experiment(eval_runs)
           
        results.append(result)
        
    pth = os.path.join('.', "results.json")
    with open(pth, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Stored results in: {pth}")
    
if __name__ == "__main__":
    main()