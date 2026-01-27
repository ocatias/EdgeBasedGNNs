"""
To add an experiment to the queue.
This does not actually run the experiment.

The idea is that you can use queue to start multiple different experiments that are performed via agents that run auto.py
"""

import argparse
import random
import time
import string
import yaml

import wandb

def parse(passed_args = None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    
    # Sweep ID, to restart an existing sweep (STILL NEEDS THE CORRECT COMMANDLINE PARAMETERS)
    parser.add_argument("--sweep", type=str, default="None", help="Sweep ID, to restart an existing sweep")
    
    # Experiment parameters
    parser.add_argument("-code", type=str, help="Path to code; E.g.: Example/run_model.py")
    parser.add_argument("-config", type=str, help="Config that contains the hyperparameter grid")
    parser.add_argument("-repeats", type=int, default=10, help="Number of times the final model will be trained+evaluated")
    parser.add_argument("-name", type=str, default="", help="Name under which the results will be stored (If this is not set, the sweep_id will be used)")
    parser.add_argument("-metric", type=str, help="Metric to optimize for, as given to WandB. E.g. Final/Val/Error")
    parser.add_argument("-min", type=bool, help="Whether the metric should be minimized (True) or maximized (False)")
    parser.add_argument("-project", type=str, help="WandB project name; Default: UmibozuTest")


    # Filter for known an unknown args (i.e. args that start with --*)

    if passed_args is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(passed_args)

    dynamic_params = []
    for i in range(len(unknown)):
        if unknown[i].startswith('--*'):
            key = unknown[i][3:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--*'):
                value = unknown[i + 1]
                dynamic_params.append(key)
                setattr(args, key, value)
    
    return args, dynamic_params

def smart_experiment_naming(parameters_dict):
    """
    Ideally the name will be umi_$DATASET_$MODEL_$RANDOMSTRING
    Tries to automatically infer dataset and model. 
    """
    
    name = ""

    def add_to_name_if_exists_and_is_unique(key, name):
        if key in parameters_dict and "value" in  parameters_dict[key]:
            name += "_" + str(parameters_dict[key]["value"])
        return name

    name = add_to_name_if_exists_and_is_unique("dataset", name)
    name = add_to_name_if_exists_and_is_unique("model", name)
    
    return f"umi" + name + "_" + ''.join(random.choices(string.ascii_letters + string.digits, k=16))

def queue_experiment(args, dynamic_params):
    assert args.sweep == "None", "Sweep ID is not a valid argument for queue.py (only for benchmark.py)"
    
    # Seed for the sweep name        
    random.seed(int(time.time()))

    # Load config parameters
    with open(args.config) as file:
        parameters_dict =  yaml.safe_load(file)
        
    # Todo: missing paramters
    for param in dynamic_params:
        parameters_dict.update({
        param: {
            'value': args.__dict__[param]}
        })

    # Create config
    sweep_config = {
        'method': 'grid',
        'name': smart_experiment_naming(parameters_dict),
        'description': 
            str({
                "--config": args.config,
                "--repeats": args.repeats, 
                "--name": args.name,
                "-min": args.min,
                "-metric": args.metric,
                "-code": args.code
            }),
        'metric': 
            {
            'name': args.metric,
            'goal': "minimize" if args.min else "maximize"   
            }
        }
    
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    
    return sweep_id


if __name__ == "__main__":
    queue_experiment(*parse())