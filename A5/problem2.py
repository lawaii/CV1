import torch.nn as nn

class CNN(nn.Module):
    """
    Constructs a convolutional neural network according to the architecture in the exercise sheet using the layers in torch.nn.

    Args:
        num_classes: Integer stating the number of classes to be classified.
    """
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        #
        # You code here
        #
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Update input channels to 1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully Connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)  # Output size adjusted for binary classification
        )

        # Softmax layer for classification
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = self.softmax(x)  # Apply softmax for classification
        return x


class LossMeter(object):
    """
    Constructs a loss running meter containing the methods reset, update and get_score. 
    Reset sets the loss and step to 0. Update adds up the loss of the current batch and updates the step.
    get_score returns the runnung loss.

    """

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss = 0.
        self.step = 0.

    def update(self, loss):
        self.loss += loss
        self.step += 1

    def get_score(self):
        return self.loss / self.step


def analysis():
    """
    Compare the performance of the two networks in Problem 1 and Problem 2 and
    briefly summarize which one performs better and why.
    """
    print("YOUR ANALYSIS")
    print("The model in Problem 2 is better. Compared with Problem 1, it has some extra layers such as convolution "
          "and max-pooling which improved the model in different aspects."
          "For example, the convolution layer automatically learns and extracts relevant features from the input data, "
          "and the max-pooling layer reduces the spatial dimensions of the feature map, which helps to control "
          "overfitting and computational complexity while preserving the most important features. That's why the "
          "model in Problem 2 has better evaluation metrics such as accuracy and loss than the former one.")