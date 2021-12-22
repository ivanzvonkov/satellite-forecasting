# Satellite Forecasting

This repository is based on the [repository](https://github.com/shgaurav1/DVG) for [Diverse Video Generation using a Gaussian Process Trigger](https://openreview.net/forum?id=Qm7R_SdqTpT) published by [Gaurav Shrivastava](https://www.cs.umd.edu/~gauravsh/) and [Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/). 

## Environment
Ensure you have [anaconda](https://www.anaconda.com/download/) installed and run:
```bash
conda env create -f environment.yml # Creates environment
conda activate dvg                  # Activates environment
```

## Arizona Dataset
Dataset (~100GB) is available in the cluster and is available for access if requested.  Please contact hkjoo@cs.umd.edu for permission request.

## Training the model
To train the DVG model on 64x64 Arizona datasets, run:
```
python DVG.py --data_root /path/to/Arizona/dataset --dataset satellite --home_dir /save/path/for/generated/outputs
```

If there is a memory error, please lower the batch size by adding the following parameter: --batch_size \<default:50\>

Please update the parameters if you would like to run the model with a different configuration.
