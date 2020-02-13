# A1: Constraint-based verification

## Requirements
Make sure Python3.6 and above is installed

To install Pytorch
```
pip3 install torch torchvision
```

To install Z3
```
pip install z3-solver
```

To run the program
```
python3 feed-forward-nn.py
```

## Goals

This assignment is designed so that 
1. You are comfortable with PyTorch and training simple neural networks. You will have to do a lot of digging to understand parts of the code; don't worry, this is part of the assignment.
2. You are comfortable with encoding neural networks and correctness properties as formulas. You will have to do a lot of digging to understand Z3's API.

## The code

- The class ``` NeuralNet``` defines the neural network. It's a very simple neural network, with a linear layer followed by ReLUs 
