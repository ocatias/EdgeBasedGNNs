# Umibozu

A light weight machine learning orchestration system for parallelized hyperparameter sweeps and model evaulation. Designed to make research faster and to be easily incorporated into existing projects (if they use [Weights&Biases](https://wandb.ai/)).

**Use case:** You want to tune a machine learning model on different datasets and evaluate the best hyperparameter multiple times. Doing this would take a long time unless you distribute these tasks across different machines. Umibozu automates all of this for you. 


## Setup

1. Copy all Umibozu files into your project.

2. Ensure that you have a script that runs your model with a function: `train_model(passed_args)` (see `Example/run_model.py`) that:
    - Takes as input a parsed list of commandline arguments
    - Tracks its results in a WandB run
    - Sets random seeds and takes them as an argument
    - Does **not** initialize a WandB run (Umibozu will do this for you)

3. Add the name of your WandB project (that the file from 2 writes to) into `config.yaml`.
```
echo '"wandb_project": $YOURPROJECTNAMEHERE' > Umibozu/config.yaml
```

4. Done. You can now use Umibozu.


## Example

1. To initialize an experiment (this does not yet run the experiment): 
```
python q.py -code Example/run_model.py -repeats 5 -name UmibozuTest01 -metric Final/Val/Error -min True --project UmibozuTest --*dataset ZINC --*model GIN -config Example/hyperparams_grid.yaml
```
This will tune the hyperparameters from the grid defined in `Example/hyperparams_grid.yaml` on the `ZINC` dataset for the `GIN` model. The goal is to `minimize` the `Final/Val/Error` metric. After finding the hyperparameter that minimizes that metric we will train `5` models with this hyperparameter combination on different seeds. 
 
2. To run this experiment, simply call 
```
python auto.py
``` 

This will automatically have the server work on an available experiment. You can also run `auto.py` on different servers. As long as they are all on the same code version they will all be running experiments.

3. To collect results:
```
python collect_results.py
```
This will create a file `results.json` that contains all information about your Umibozu run. You can turn your results directly into a table with:
```
python collect_results.py --key "Final/Test/mae" --row model --column dataset
```

This creates a table with models as rows and columns as different datasets. Each entries is the mean and standard deviation of `Final/Test/mae` of the evaluation.


## Documentation

### What does `--*`do?
Parameters that begin with `--*` can be used to set commandline parameters from `train_model` without explicitly writting them into the grid (this means you do not need separate grids for different datasets as long as the other hyperparameters are the same). Note that `q.py` and `benchmark.py` do not need any changes to support `--*` parameters. If your `train_model` uses any specific parameters, they will work directly with Umibozu!


### How do I stop an experiment?
If you are still tuning hyperparameters, pause the WandB sweep. If you are in evaluation you will either have to delete the WandB sweep or edit its description such that repeats is set to 0 (in both cases, you need to manually shut down the the agents).

### What do I do if a run crashes?
They should automatically restart as long as `auto.py` or `auto.sh` are running.
