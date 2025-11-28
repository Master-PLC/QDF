# Quadratic Direct Forecast (QDF)

Official implementation for "Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models"

## Reproduction

You can reproduce the results by running the provided shell scripts:

```bash
bash ./scripts/ETTh1.sh
bash ./scripts/ETTh2.sh
bash ./scripts/ETTm1.sh
bash ./scripts/ETTm2.sh
bash ./scripts/ECL.sh
bash ./scripts/Weather.sh
bash ./scripts/PEMS03.sh
bash ./scripts/PEMS08.sh

# Or run all datasets with
bash ./scripts/run_all.sh
```
All the hyper parameters used in the paper are included in these scripts.

## Components

The major components of QDF—including meta-train, meta-test, and inner loop algorithms—are implemented in [`exp/exp_long_term_forecasting_meta_ml3.py`](./exp/exp_long_term_forecasting_meta_ml3.py).

- **Meta-train phase**  
The meta-training phase is responsible for learning a task-adaptive loss function, which is achieved by training with multiple meta-tasks. See the method and related logic around meta training in the class (starting in the `train()` method and calls to `meta_train(...)`, typically after data preparation).

- **Meta-test phase**  
The meta-testing phase evaluates transferability and generalization using the learned loss. In this phase (see `meta_test(...)` and its use in `train()`), the model is retrained using the learned loss function, then validated for performance. For details, see `lines 381-436`.

- **Inner loop**  
Each meta-task iteratively updates model parameters using the learned loss function. This inner loop mimics a few-shot adaptation setup and is crucial for learning meta-parameters. The implementation can be found in the `inner_loop()` method. For details, see `lines 159-232`.

- **Loss and covariance structure**  
The core mathematical novelty (covariance loss, etc) is implemented in the `CovarianceMatrix` class.


## Environment

All required dependencies are listed in `requirements.txt`.  
To set up the environment, run:

```bash
sudo apt install bc

conda create -n qdf python=3.10
conda activate qdf
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Acknowledgements

The implementaiton of this project is built upon some established repos, such as FreDF and Time-o1.