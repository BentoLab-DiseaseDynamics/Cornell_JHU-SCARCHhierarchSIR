# Cornell_JHU-SCARCHhierarchSIR

A hybrid SIR - Bayesian model for infectious disease forecasting. Successor to Cornell_JHU-hierarchSIR.

## Installation (local)

Available platforms: macOS and Linux.

### Setup and activate a conda environment

Update conda to make sure your version is up-to-date,

```
conda update conda
```

Setup/update the `environment`: All dependencies needed to run the scripts are collected in the conda `SCARCHhierarchSIR_env.yml` file. To set up the environment,

```
conda env create -f SCARCHhierarchSIR_env.yml
conda activate BENTOLAB-SCARCH_HIERARCHSIR
```

or alternatively, to update the environment (needed after adding a dependency),

```
conda activate BENTOLAB-SCARCH_HIERARCHSIR
conda env update -f SCARCHhierarchSIR_env.yml --prune
```

### Install the `SCARCHhierarchSIR` package

Install the `SCARCHhierarchSIR` Python package inside the conda environment using,

```
conda activate BENTOLAB-SCARCH_HIERARCHSIR
pip install -e . --force-reinstall
```

### Model training and forecasting

#### Training (execute once at season start)


#### Forecast (performed automatically using GH actions)

## Training on a cluster

See ...

## Workflows
