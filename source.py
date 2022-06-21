import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

import matplotlib.pyplot as plt

train_set = datasets.MNIST("data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_set = datasets.MNIST("data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

#######################################################
# THE TRAINING AND TESTING IMAGES ARE ACTUALLY INVERTED!
# THE WRITING IS IN WHITE WHILE THE BACKGROUND IS BLACK!
########################################################

# We are planning to train our model in batches.
BATCH_SIZE = 16
train_batch_set = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

PIXEL_COUNT = 28 * 28
HIDDEN_LAYER_1_NEURON_COUNT = 128
HIDDEN_LAYER_2_NEURON_COUNT = 64
HIDDEN_LAYER_3_NEURON_COUNT = 32
OUTPUT_LAYER_NEURON_COUNT = 10

model = nn.Sequential(
    nn.Linear(PIXEL_COUNT, HIDDEN_LAYER_1_NEURON_COUNT),
    nn.ReLU(),
    nn.Linear(HIDDEN_LAYER_1_NEURON_COUNT, HIDDEN_LAYER_2_NEURON_COUNT),
    nn.ReLU(),
    nn.Linear(HIDDEN_LAYER_2_NEURON_COUNT, HIDDEN_LAYER_3_NEURON_COUNT),
    nn.ReLU(),
    nn.Linear(HIDDEN_LAYER_3_NEURON_COUNT, OUTPUT_LAYER_NEURON_COUNT),
    nn.LogSoftmax(dim=1)
)

# Get the next item in an iterator (train_batch_set). In our case the next
# item is a batch of training data.
batch = next(iter(train_batch_set))

# Get the first item in the batch and flatten it into size 1 x PIXEL_COUNT
x = batch[0][0]
x = x.view(1, PIXEL_COUNT)

# Check to see if our model can feed forward this x (input data). The output
# y_hat is a random guess as our model is not currently trained.
y_hat = model(x)
print(y_hat.argmax())

# Training our model
EPOCH_COUNT = 100
LR = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training!
for _ in range(EPOCH_COUNT):

    # Doing batch gradient descent
    for batch_data, batch_label in train_batch_set:

        # Reset the current gradient to 0.
        optimizer.zero_grad()

        # Run inference on our current model.
        batch_output = model(batch_data.view(BATCH_SIZE, PIXEL_COUNT))

        # Calculate the inference loss of our current model.
        loss = F.nll_loss(batch_output, batch_label)

        # optimizer.step is performs a parameter update based on the current gradient
        # (stored in .grad attribute of a parameter) and the update rule.

        # Calling .backward() multiple times accumulates the gradient (by addition) for
        # each parameter. This is why you should call optimizer.zero_grad() after each
        # .step() call. Note that following the first .backward call, a second call is
        # only possible after you have performed another forward pass.

        # Propagate the loss back into our neural net.
        loss.backward()

        # Do gradient descent to minimize loss.
        optimizer.step()

# Testing our model: Inference
correct_count = 0
for x, y in test_set:
    y_hat = model(x.view(1, PIXEL_COUNT))
    # print("Predict: ", y_hat.argmax(), ". Truth: ", y)
    if y_hat.argmax() == y:
        correct_count += 1

print("Accuracy: ", correct_count * 100 / len(test_set))

# Save model for future use
torch.save(model, "09-30-2021")
