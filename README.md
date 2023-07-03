tetris-deep-q-learning

# Overview

<img align="right" src="tetris_gameplay.gif">

This repo uses deep reinforcement learning to learn a policy to play tetris. A deep Q network (DQNN) is implemented using PyTorch and OpenAI gym. The trained model can consistently clear ~1000 lines, but further training and different networks architecture/hyperparameters can clear even more.

# Installation
Install the latest version of Anaconda for your system from [here](https://docs.anaconda.com/anaconda/install/), then execute the following steps to setup the repository and install the required dependencies.
```
git clone https://github.com/SchultzKyle/tetris-deep-q-learning.git
cd tetris-deep-q-learning
conda env update -n tetris-deep-q-learning -f environment.yml
conda tetris-deep-q-learning
```

# Usage
The main files are:

1. player.py, runs the learned policy and reports statitics
2. train.py, trains the DQNN from scratch.
3. trained_model, the trained network output by train.py
4. dqnn.py, specifies the DQNN architecture
5. envs/tetris.py, simulates the game of tetris as an OpenAI gym environment

