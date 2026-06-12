# Recommender Systems assignment 4 - LightGCN

This repository contains an implementation of LightGCN on PyTorch. LightGCN is a graph-based recommender algorithm that significantly simplifies graph operations by interpolating across increasingly farther neighbours' preferences. Additionally, this version of LightGCN implements a discount factor parameter $\gamma$ (inspired by reinforcement learning agorithms), which "discounts" further neighbours compared to closer ones.

The data for this project can be found <a href="https://www.kaggle.com/competitions/recommender-system-challenge-2526-s-2/data">here</a>. The data contains user-item interactions and metadata for the items. Please, place the datasets in a folder named `data` inside the main project directory, outside of `src`.

## Setting up

To install necessary packages in your environment, navigate to the root folder of the project and run:

```{bash}
pip install -r requirements.txt
```

## How to run

In order to run the training of the LightGCN model, navigate to the main project directory and run the following code:

```{bash}
python src/train_lightgcn.py
```

The training will run for 100 epochs. Upon the end of the training, images for the loss progression and the evaluation metrics will be saved as PNGs in the folder `images/`. During training, each time the Recall@10 improves, the weights of the model are saved at `checkpoints/best_lightgcn.pt`.

### Tuning

The hyperparameters stored in `config.py` are already tuned on Recall@10. However, should you want to re-tune them, you can run the following script:

```{bash}
python src/lightgcn_tuning.py
```

The tuning will run for 50 trials using optimised grid search by Optuna. You can follow the trials in real time by running in a separate terminal the following:

```{bash}
optuna-dashboard sqlite:///db.sqlite3
```

This will provide a `localhost` link. When you click on it, a dashboard will open with the metrics per trial. At the end of the tuning, a CSV file with the name of the experiment containing the trials' information will be saved to `tuning/`.

## Contact

For questions, please contact e.roncaglia@umail.leidenuniv.nl.
