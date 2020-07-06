import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from z3 import (
    Solver,
    RealVector,
    Sum,
    If,
    Or,
    is_algebraic_value,
    is_rational_value,
    is_int_value)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting the random seed for uniformity
torch.manual_seed(2)

PATH = './data'


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # initializes a weight matrix of shape (hidden_size * input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        # initializes a weight matrix of shape (num_classes * hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    # invoke all functions in order to compute the final outcome/digit class
    # in a forward pass
    def forward(self, x):
        # applies linear transformation to the input data, y = wx + b
        out = self.fc1(x)
        # applies the RELU activation function on the hidden layer
        out = self.relu(out)

        # applies linear transformation to the hidden layer to map to the
        # output layer
        out = self.fc2(out)
        return out


def init_model_parameters():
    global num_epochs, batch_size, model, criterion, optimizer

    # Hyper-parameters
    # The dataset contains gray-scale images of pixel dimensions 28*28

    # Input layer contains 784 nodes, one node for each pixel in the image
    # Hidden layer contains 500 nodes
    # Output layer contains 10 nodes (one for each class representing a digit
    # between 0-9)
    input_size = 784
    hidden_size = 10
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    # Cross Entropy loss for the predicted outcome vs actual result is
    # evaluated in the training phase after the forward pass
    criterion = nn.CrossEntropyLoss()
    # Adam stochastic optimization algorithm implements an adaptive learning
    # rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load the MNIST dataset


def load_dataset():
    global train_dataset, test_dataset, train_loader, test_loader

    # MNIST dataset contains 60000 images in the training data and 10000 test
    # data images
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader divides the dataset into batches of batch_size=100 that can
    # be used for parallel computation on multi-processors
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


def save_model():
    torch.save(model.state_dict(), PATH)


def load_model():
    model.load_state_dict(torch.load(PATH))
    model.eval()


def train():
    total_step = len(train_loader)
    # Iterate over all training data (600 images in each of the 100 batches)
    # in every epoch(5)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # manually set the gradients to zero
            optimizer.zero_grad()
            # compute the new gradients based on the loss likelihood
            loss.backward()
            # propagate the new gradients back into NN parameters
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        torch.save(model.state_dict(), 'model.ckpt')


def test():
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            # pass the test images batch to the trained model to compute
            # outputs
            outputs = model(images)
            # fetching the class with maximum probability for every image in
            # the batch as the predicted label
            _, predicted = torch.max(outputs.data, 1)
            # print("Bing: ", outputs, predicted)
            total += labels.size(0)
            # compute the total correctly predicted outcomes (when test image
            # label = predicted)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(
            100 * correct / total))


# fetch the weights of the neural network returned as tensors
# pass layer as int (1: input to hidden or 2: hidden to output)
def fetch_weights(layer):
    # Weights between input layer and hidden layer (Tensor of shape: [#hidden,
    # 784] i.e. hidden layer size * input layer size)
    if layer == 1:
        return model.fc1.weight.data
    # Weights between input layer and hidden layer (Tensor of shape: [10,
    # #hidden] i.e. output layer size * hidden layer size)
    if layer == 2:
        return model.fc2.weight.data


# Plot the first image from a dataset
def plot_dataset(image):
    plt.imshow(image[0][0], cmap='gray')
    plt.show()


# Plot a pair of original and altered image
def plot_pair(image1, image2):
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image1[0], cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(image2[0], cmap='gray')
    plt.show()


# Plot all pairs of images
# shape of images should be [n][2][x][y]
# n pairs, 2 in a pair, x & y are pixel sizes
def show_pairs(images):
    n = images.size()[0]
    fig = plt.figure()
    for i in range(n):
        fig.add_subplot(n, 2, 2 * i + 1)
        plt.imshow(images[i][0], cmap='gray')
        fig.add_subplot(n, 2, 2 * i + 2)
        plt.imshow(images[i][1], cmap='gray')
    plt.show()


def verify(max_pixels):
    to_check = 20  # Number of images to verify
    checked = 0
    verified = 0
    failed = 0
    # Output counter-example pairs
    output_images = torch.zeros((to_check, 2, 28, 28))

    # Start at -1 because we increment first
    i = -1
    # Loop until we verify to_check images
    while checked < to_check:
        i += 1
        image = test_dataset[i][0]
        x = image.reshape(-1, 28 * 28).to(device)
        y = test_dataset[i][1]

        # Check if the model was correct on this example
        with torch.no_grad():
            act_y = model(x)[0].argmax()
            if y == act_y:
                checked += 1
            else:
                # If the model didn't give the right output, don't check
                # robustness
                continue

        # Create a verifier with up to the maximum variable pixels, and
        # check it. Increment by 10 for speed
        pixels_to_check = [1] + list(range(10, max_pixels + 1, 10))
        for pixels in pixels_to_check:
            verifier = Verifier(pixels)
            counter_example = verifier.solve_with(x, y)

            if not isinstance(counter_example, str):
                print('image %d is not robust for %d pixels' % (i, pixels))
                output_images[failed][0] = image
                output_images[failed][1] = counter_example.reshape(
                    -1, 28, 28).to(device)
                failed += 1
                break
            elif counter_example == 'timeout':
                print(
                    'image %d may be robust (timeout) for %d pixels' %
                    (i, pixels))
                # After a timeout, assume we can't viably check for larger
                # #pixels
                break

        # If we tried up until the max pixels and it's still unsat,
        # the image/network is robust
        if counter_example == 'unsat':
            print('image %d is robust for %d pixels' % (i, pixels))
            verified += 1

    print('Verified %d/%d are robust' % (verified, to_check))
    print('\t%d/%d are not robust' % (failed, to_check))
    print('\t%d/%d are timed out (robustness unknown)' %
          (to_check - verified - failed, to_check))

    # Show all the counter-exampled pairs
    show_pairs(output_images[:failed])


def main():
    # Display usage
    if len(sys.argv) < 2:
        print('Usage:')
        print('python cs839_hw1.py train/load[ max]')
        print('Required argument: train/load')
        print('\tSpecify train to retrain the model')
        print('\tSpecify load to load the last trained model')
        print('Optional argument: max')
        print('\tMax number of varying pixels')
        sys.exit()

    init_model_parameters()
    load_dataset()
    if sys.argv[1] == 'train':
        train()
        save_model()
    elif sys.argv[1] == 'load':
        load_model()
    else:
        print('Invalid first argument [%s], exiting...' % sys.argv[1])
        sys.exit()

    # Test the output
    test()
    # Verify with requested variable pixels (default 60)
    max_pixels = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    verify(max_pixels)


def model_to_val(m, var):
    p = m[var]
    if is_int_value(p):
        return float(p.as_long())

    if is_algebraic_value(p):
        p = p.approx(5)  # Precise to 5 decimals
    if is_rational_value(p):
        x = float(p.numerator_as_long()) / \
            float(p.denominator_as_long())
    return x


class Verifier():
    def __init__(self, pixels_to_change):
        self.solver = Solver()
        # Limit the checking to 120 seconds
        self.solver.set(timeout=120 * 1000)
        self.pixels_to_change = pixels_to_change

    def solve_with(self, input_image, output_index):
        x = RealVector('x', len(input_image[0]))
        for i in range(len(input_image[0])):
            if i >= self.pixels_to_change:
                # x[i] = image[i]
                self.solver.add(x[i] == input_image[0][i].item())
            else:
                # 0 <= x[i] <= 1
                self.solver.add(x[i] >= 0)
                self.solver.add(x[i] <= 1)

        fc1_weights = fetch_weights(1)
        fc1_shape = fc1_weights.size()
        # o1 = fc1^T * x
        o1 = [Sum([fc1_weights[i, j].item() * x[j]
                   for j in range(fc1_shape[1])]) for i in range(fc1_shape[0])]

        # y1 = ReLU(o1)
        y1 = [If(o1[i] > 0, o1[i], 0) for i in range(fc1_shape[0])]

        fc2_weights = fetch_weights(2)
        fc2_shape = fc2_weights.size()
        # y2 = fc2^T * y1
        y2 = [Sum([fc2_weights[i, j].item() * y1[j]
                   for j in range(fc2_shape[1])]) for i in range(fc2_shape[0])]

        # If any y2 output is higher than the expected y2,
        # the model output changes
        self.solver.add(Or([y2[output_index] < y2[i]
                            for i in range(len(y2)) if i != output_index]))
        # self.solver.add(And([y2[output_index] > y2[i]
        #                      for i in range(len(y2)) if i != output_index]))

        # Check if the classification can change
        check = self.solver.check()
        sat = str(check) == 'sat'

        if sat:
            m = self.solver.model()
            # Substitute the model back in
            x_new = input_image.clone().detach()
            for i in range(self.pixels_to_change):
                x_new[0][i] = model_to_val(m, x[i])
            for i in range(self.pixels_to_change, len(input_image[0])):
                x_new[0][i] = input_image[0][i]
            return x_new
        elif str(check) == 'unknown':
            return 'timeout'
        else:
            return 'unsat'


if __name__ == '__main__':
    main()
