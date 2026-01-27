"""
An example of how a script to train your model could look like.
This is only a mock-up to show how training interacts with Umibozu.

You can set the parameters from the commandline:

python run_model.py --dataset ZINC --model GCN --num_layers 4
This trains a 4 layer GCN on ZINC.
"""

import argparse
import time

import numpy as np
import wandb

wandb_project_name = "UmibozuTest"

def parse_args(passed_args = None):
    parser = argparse.ArgumentParser(description='An experiment.')

    # Your training script must support setting the seed for Umibozu to work
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    parser.add_argument('--model', type=str, default='GIN', help='Model to use (default: GIN; options: GIN, GCN, GAT)')
    parser.add_argument('--dataset', type=str, default="ZINC", help='Dataset name (default: ZINC; other options: molhiv, QM9)')
    # Some model hyparameters (not required for Umibozu)
    parser.add_argument('--num_layers', type=int, default=5, help='Number of message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=64, help='Dimensionality of hidden units (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')

    if passed_args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(passed_args)


def main(args):
    """
    Method to train a model given some arguments.
    NOTE: We initialize WandB separately so that train_model can be in charge of initializing it
    """
    
    # Trainig loop
    val_error_ls, test_error_ls = [], []
    for epoch in range(0, args.epochs):

        # Simple mock-up for how train errors can look like
        train_error = 1 + np.cos(np.pi *(float(epoch+1)/args.epochs))
        val_error = 1 + np.cos(np.pi *(float(epoch+1)/(args.epochs - args.num_layers))) + args.num_layers / args.emb_dim
        test_error = 1 + np.cos(np.pi *(float(epoch+1)/(args.epochs - args.num_layers))) + args.num_layers / args.emb_dim

        val_error_ls.append(val_error)
        test_error_ls.append(test_error)

        wandb.log({
            'Epoch': epoch,
            'Train/Error': train_error,
            'Val/Error': val_error,
            'Test/Error': test_error,
        })

        print(f"Epoch: {epoch}, Train: {train_error:.3f} Val: {val_error:.3f} Test: {test_error:.3f}", end="\r")
        time.sleep(0.02)
    
    # Report test metric in epoch with lowest validation error
    best_epoch = np.argmin(val_error_ls)    

    wandb.log({
            'Final/Val/Error': val_error_ls[best_epoch],
            'Final/Test/Error': test_error_ls[best_epoch],
        })
    
    print(f"\nBest Epoch: {best_epoch}, Val {val_error_ls[best_epoch]:.3f}, Test {test_error_ls[best_epoch]:.3f}")
    wandb.finish()


def train_model(wandb_config = None, mark_preempting = True, tag = None):
    """
    INTERFACE FOR UMIBOZU
    A function with this name is required for Umibozu to work.
    Umibozu will use this as an interface to train models
    """
    
    with wandb.init(config=wandb_config, project=wandb_project_name) as run:

        # run = wandb.init(config=wandb_config, project=wandb_project_name)
        if wandb_config is None:
            wandb_config = wandb.config

        if tag is not None:
            wandb.run.tags = [tag] 

        if mark_preempting:
            print("Marking as preempting")
            run.mark_preempting()
        
        # Change a dictionary to commandline parameter shape
        # E.g. {"model": "GCN"} -> {"--model": GCN}
        config_ls = []
        for key in wandb_config.keys():
            config_ls += [f"--{key}", str(wandb_config[key])]
        
        print(config_ls)

        args = parse_args(config_ls)
        main(args)

if __name__ == "__main__": 
    """
    COMMANDLINE INTERFACE FOR THE USER
    """

    args = parse_args()

    wandb.init(
        config = args,
        project = wandb_project_name
        )
    
    main(args)