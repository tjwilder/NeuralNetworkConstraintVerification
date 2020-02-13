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

- This is the (at this point in time) toy classification task for MNIST digits. Each image is a handwritten digit between 0 and 9 and our goal is to classify the digit. Each image is 768 pixels (28x28).

- The class ``` NeuralNet``` defines the neural network. It's a very simple neural network, with a linear layer (10 nodes) followed by ReLUs (10), followed by a linear layer (10). Therefore, the neural network has 10 outputs, indicating which digit we are talking about. (Note that there is no softmax layer.)

- ```train()``` trains the neural network, and ```test()``` tests it. You might want to add a function that saves the trained neural network so you do not retrain it every time.

- the function ```plot_dataset``` draws a given image; see ```main()``` for an example.

## Your job

Your job is to, given an image I with label L, generate a set of constraints that checks whether classification changes if the first N pixels are modified arbitrarily. Start with N=1, and go up until verification is too slow or you can always change the classification. Try to verify robustness of ~20 images from the test set whose labels are predicted correctly.

Following our notation from class, this is the property we want for an image I with label L

```
{x is like I but the first N pixels are different}
r <- f(x)
{argmax_i r_i = L}
```
Notice that we take the largest index of the size 10 output vector. 

If you find verification is too slow, you can make the network smaller, e.g., by making the hidden layer smaller or by removing the last linear layer.

If verification fails, the SMT solver will give you a model, which is an image. Make sure to print out the image. This will entail converting the model back into a tensor.

Comment your code extensively. This is not an undergraduate assignment where we run your code through a testing suite. We will look at your encoding to see if you understand the problem and 
