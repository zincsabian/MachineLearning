import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import singleton
import module
import torch.optim as optim

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

@singleton
def load_data():    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
                                    './data/', 
                                    train=True, 
                                    download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), 
                                        (0.3081,)
                                        )
                                    ])),
        batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
                                    './data/', 
                                    train=False, 
                                    download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), 
                                        (0.3081,)
                                        )
                                    ])),
        batch_size=batch_size_test, shuffle=True)
    
    return train_loader, test_loader


def show_data():
    train_loader, test_loader = load_data()
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(batch_idx)
    print(example_data.shape)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
        plt.show()

show_data()
network = module.CNN()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)