# Reward Estimation for Variance Reduction in Deep Reinforcement Learning
## Installation

We based our code primarily off of [ikostrikov's pytorch-rl repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr). Follow installation instructions there.

### Make sure to install pytorch 0.3.1 (ikostrikov's repo is already using version 0.4.0 - which is incompatible with this code base)

## How to run

To replicate the mujoco results (with gaussian noise) from the paper you need to run all 750 runs individually with:

``` python main.py --continuous --use-gaussian-noise --run-index [0-749] ```

To replicate the mujoco results (with uniform noise) from the paper you need to run all 750 runs individually with:

``` python main.py --continuous --use-uniform-noise --run-index [0-749] ```

To replicate the mujoco results (with sparse noise) from the paper you need to run all 750 runs individually with:

``` python main.py --continuous --use-sparse-noise --run-index [0-749] ```

To replicate the atari results (with gaussian noise) from the paper you need to run all 270 runs individually with:

``` python main.py --use-gaussian-noise --run-index [0-269] ```

To replicate the atari results (with uniform noise) from the paper you need to run all 189 runs individually with:

``` python main.py --use-uniform-noise --run-index [0-188] ```

To replicate the atari results (with sparse noise) from the paper you need to run all 189 runs individually with:

``` python main.py --use-sparse-noise --run-index [0-188] ```

## Visualization

run visualize.py to visualize performance (requires Visdom)

## Citation

If you find this useful, please cite our work:


```
@inproceedings{hendersonromoff2018optimizer,
  author    = {Joshua Romoff and Peter Henderson and Alexandre Piche and Vincent Francois-Lavet and Joelle Pineau},
  title     = {Reward Estimation for Variance Reduction in Deep Reinforcement Learning},
  booktitle = {Proceedings of the 2nd Annual Conference on Robot Learning(CORL 2018)},
  year      = {2018}
}

```

Additionally, if you are relying on the codebase heavily please note the original codebase as well:

```
@misc{pytorchrl,
  author = {Kostrikov, Ilya},
  title = {PyTorch Implementations of Reinforcement Learning Algorithms},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr}},
}
```

## License
This repo is CC-BY-NC licensed, as found in the LICENSE file.
