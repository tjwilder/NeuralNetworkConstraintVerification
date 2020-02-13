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

- This is the, at this point, toy classification task for MNIST digits. Each image is a handwritten digit between 0 and 9 and our goal is to classify the digit. Each image is 768 pixels (28x28).

- The class ``` NeuralNet``` defines the neural network. It's a very simple neural network, with a linear layer (10 nodes) followed by ReLUs (10), followed by a linear layer (10). Therefore, the neural network has 10 outputs, indicating which digit we are talking about. (Note that there is no softmax layer.)

- ```train()``` trains the neural network, and ```test()``` tests it. You might want to add a function that saves the trained neural network so you do not retrain it every time.

- the function ```plot_dataset``` draws a given image; see ```main()``` for an example.
