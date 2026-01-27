"""
Performs hyperparameter tuning and evaluation for a GNN architecture on a given dataset + task

If sweep ID is given: runs the corresponding experiment

If no sweep ID is given: starts experiment with queue.py and then performs it
"""
import wandb
import time
from datetime import datetime
import importlib

from q import parse, queue_experiment

disable_auto_safeguards = False

def code_path_to_fct(path):

    # Reformat paths that look like Example/run_model.py into Example.run_model
    if path[-3:] == ".py":
        path = path[:-3]

    path = path.replace('/', '.')

    mod = importlib.import_module(path)

    return getattr(mod, 'train_model')
     

def get_eval_runs(best_tune_run, eval_tag, include_running_runs = False):
    # Make a seperate API so we can find new runs as well
    api = wandb.Api()
    runs = api.runs(f"{best_tune_run.entity}/{best_tune_run.project}")
    
    # Run needs to have the eval_tag
    runs_with_eval_tag = [run for run in runs if eval_tag in run.tags]
    
    # Only include finished runs
    if include_running_runs:
        finished_runs = [run for run in runs_with_eval_tag if run.state.lower() in ["finished", "running"]]
    else:
        finished_runs = [run for run in runs_with_eval_tag if run.state == "finished"]
        
    return finished_runs

def main(passed_args = None):       
    args, _ = parse(passed_args)
    
    api = wandb.Api()
    
    # New sweep
    if args.sweep == "None":
        sweep_id = queue_experiment(args)
        wandb.agent(sweep_id, code_path_to_fct(args.code))
        sweep = api.sweep(args.project + "/" + sweep_id)

    # Sweep already exists
    else:
        sweep_id = args.sweep
        sweep = api.sweep(args.project + "/" + sweep_id)
        if sweep.state.lower() == "finished":
            pass
        elif sweep.state.lower() == "paused":
            raise Exception("SWEEP IS PAUSED")
        elif sweep.state.lower() in ["running", "pending"] :
            wandb.agent(args.project + "/" + sweep_id, code_path_to_fct(args.code))
        else:
            raise Exception(f"Unknown sweep state: {sweep.state}")
        
    # Check if sweep is done 
    if sweep.state.lower() != "finished":
        # Sweep has been canceled -> stop
        print(f"Sweep state: {sweep.state.lower()}.")
        return
    
    # Check if all hyperparameter configs are done (otw we might have just deleted a pre-empted run and stil need to re-run it)
    else:
        num_runs = len(list(sweep.runs))

        # Get number of hyperparam configurations
        params = sweep.config["parameters"]
        num_hyp_configs = 1
        for param in params.values():
            if "values" in param:
                num_hyp_configs *= len(param["values"])

        if num_hyp_configs != num_runs and not disable_auto_safeguards:
            print(f"Not all hyperparameters have been run ({num_runs}/{num_hyp_configs}). Sweep needs to be manually restarted!")
            return    
     
    # Check if all runs are finished (if not then quit this benchmark)
    for run in list(sweep.runs):
        if run.state.lower() != "finished":
            
            # TBD if run is preempting / preempting and no updates in > 2 minutes delete it, restart sweep and queue an agent
            print(f"Sweep still has some unfinished runs: {run.state.lower()}")

            if (datetime.now().timestamp() - run.summary["_timestamp"]) > 120:
                run.delete()
                print("Pre-empted run has been deleted, but sweep still needs to be restarted manually")
                return
            else:
                return

    # Get best run
    best_run = sweep.best_run()
    best_config = best_run.config
    eval_tag = f"{sweep_id}--EVAL"
    
    print("Best run ID:", best_run.id)
    print("Best run:", best_run.summary)
    print("Best run config:", best_run.config)
        
    # This is needed so W&B does not get confused after the sweep
    wandb.teardown()
    
    # Evaluation
    for n in range(1, args.repeats + 1):
        filtered_runs = get_eval_runs(best_run, eval_tag)
        filtered_runs_include_running_runs = get_eval_runs(best_run, eval_tag, include_running_runs=True)   
        
        # Check if enough evaluations have concluded already
        if len(filtered_runs) >= args.repeats:
            break
        
        # Check if someone else is running the final evaluations
        elif len(filtered_runs_include_running_runs) >= args.repeats:
            return
        
        print(f"EVALUATION: {len(filtered_runs_include_running_runs) + 1} / {args.repeats}")
        best_config['seed'] = len(filtered_runs_include_running_runs) + 1
        (code_path_to_fct(args.code))(best_config, mark_preempting = False, tag = eval_tag)
        
    print("\n\n\nCOLLECTING FINAL RESULTS")

    # Give W&B some time to sync
    time.sleep(5)
    
if __name__ == "__main__":
    main()