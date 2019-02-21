## An Empirical study of Binary Neural Networks' Optimisation

The source code used for experiments in the paper "[An Empirical study of Binary Neural Networks' Optimisation](https://openreview.net/forum?id=rJfUCoR5KX)".

The code grew organically as we tweaked more and more hyperparameters. Had I been more familiar with class-based declerations in TensorFlow (or embraded PyTorch sooner) the code would have been more elegant.

### Environment
This code has been only tested with TensorFlow 1.8.0 and Python 3.5.4. The exact environment can be replicated by:

`$ conda env create -f environment.yml`

This would create a conda environment called `studying-bnns`.

### Usage

```bash
$ conda activate studying-bnns

# Run an experiment defined in a YAML file
$ python run_with_yaml.py --model binary_connect_mlp \
    --dataset mnist --epochs 250 --batch-size 100 \
    --binarization deterministic-binary

# Run an experiment by passing args
python run_with_args.py some_experiment.yaml
```

An example experiment defintion in YAML file:


```yaml
experiment-name: some_experiment

model: binary_connect_cnn
dataset: cifar10
epochs: 500
batch_size: 50

binarization: deterministic-binary

learning_rate:
  type: exponential-decay
  start: 3e-3
  finish: 2e-6

loss: 'square_hinge_loss'

optimiser:
  function: tf.train.AdamOptimizer
  kwargs: "{'beta1': 0.9, 'beta2': 0.999}"
```
