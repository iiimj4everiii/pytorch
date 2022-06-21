import torch
from torchvision import transforms, datasets

import matplotlib.pyplot as plt

data_transformer = transforms.Compose([transforms.ToTensor()])
test_set = datasets.MNIST("data", train=False, download=True, transform=data_transformer)

#######################################################
# THE TRAINING AND TESTING IMAGES ARE ACTUALLY INVERTED!
# THE WRITING IS IN WHITE WHILE THE BACKGROUND IS BLACK!
########################################################

model = torch.load("09-30-2021")

# Testing our model: Inference
PIXEL_COUNT = 28 * 28
correct_count = 0
for x, y in test_set:
    y_hat = model(x.view(1, PIXEL_COUNT))
    # print("Predict: ", y_hat.argmax(), ". Truth: ", y)
    if y_hat.argmax() == y:
        correct_count += 1

print("Accuracy: ", correct_count * 100 / len(test_set))

# Test my own handwriting. Transforming the images to grayscale and formatted as tensors.
# WE ALSO NEED TO INVERT THE COLOR SINCE THE TRAINING IMAGES ARE INVERTED: WRITING IN WHITE
# AND BACKGROUND IS BLACK.
data_transformer = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.RandomInvert(p=1), transforms.ToTensor()])
my_dataset = datasets.ImageFolder("my_handwriting", transform=data_transformer)

for data in my_dataset:
    y_hat = model(data[0].view(1, PIXEL_COUNT))
    print("Predict: ", y_hat.argmax())

    plt.imshow(data[0].view(28, 28), cmap='gray')
    plt.show()
print()
