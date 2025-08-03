# Setup Instructions for `nickyobia_env` Conda Environment

This guide explains how to create the `nickyobia_env` Conda environment from an `environment.yml` file and register it for use in Jupyter Notebook.

## Step 1: Create the Conda Environment

Run the following command in your terminal:

conda env create -n nickyobia_env -f environment.yml

This will:
- Install all dependencies listed in `environment.yml`
- Create a new environment named `nickyobia_env` (overriding the name specified in the file, if any)

## Step 2: Activate the Environment

conda activate nickyobia_env


## Step 3: Register the Environment with Jupyter

If `ipykernel` is not already installed:

conda install ipykernel

Then register the environment as a Jupyter kernel:

python -m ipykernel install --user --name=nickyobia_env --display-name "Python (nickyobia_env)"


## Step 4: Launch Jupyter Notebook or JupyterLab

jupyter notebook

or

jupyter lab

In the interface, select the kernel named **"Python (nickyobia_env)"**.


## Optional: Update the Environment

To update the environment if `environment.yml` changes:

conda env update -n nickyobia_env -f environment.yml --prune

## Optional: Remove the Environment

To completely remove the environment:

conda remove --name=nickyobia_env --all


## Optional: Export the Environment

To generate a new `environment.yml` from your current environment:

conda env export --no-builds > environment.yml
