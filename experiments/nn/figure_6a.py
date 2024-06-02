import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

# Import the Wide ResNet model (adjust the import based on your actual script)
from libraries.wide_resnet.networks.wide_resnet import WideResNet


# Load CIFAR-10 datasets
def load_cifar10(train_path, test_path):
    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)
    return train_dataset, test_dataset


# Add label noise
def add_label_noise(labels, noise_level, num_classes=10):
    noisy_labels = labels.clone()
    n_samples = len(labels)
    n_noisy = int(noise_level * n_samples)
    noisy_indices = random.sample(range(n_samples), n_noisy)

    for idx in noisy_indices:
        # Resample label from alternative class labels, excluding the ground truth label
        new_label = random.choice([l for l in range(num_classes) if l != labels[idx]])
        noisy_labels[idx] = new_label

    return noisy_labels


# Train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, lr_schedule):
    model.train()
    for epoch in range(epochs):
        if epoch in lr_schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


# Main experiment function
def experiment(train_sizes, noise_levels, epochs, trials, train_path, test_path, lr, momentum, batch_size):
    train_dataset, test_dataset = load_cifar10(train_path, test_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_results = {size: {noise: [] for noise in noise_levels} for size in train_sizes}

    for size in train_sizes:
        for noise in noise_levels:
            for trial in range(trials):
                subset_indices = random.sample(range(len(train_dataset)), size)
                subset = Subset(train_dataset, subset_indices)

                noisy_labels = add_label_noise(subset.targets, noise, num_classes=10)

                train_loader = DataLoader(list(zip(subset, noisy_labels)), batch_size=batch_size, shuffle=True)

                model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

                lr_schedule = [30, 40]
                _, accuracy = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs,
                                          lr_schedule)

                test_error = 1 - accuracy
                all_results[size][noise].append(test_error)

    # Averaging results
    avg_results = {
        size: {noise: (np.mean(all_results[size][noise]), np.std(all_results[size][noise])) for noise in noise_levels}
        for size in train_sizes}

    # Plotting results
    plt.figure(figsize=(12, 6))
    for size in train_sizes:
        noise_levels_list = sorted(avg_results[size].keys())
        means = [avg_results[size][noise][0] for noise in noise_levels_list]
        stds = [avg_results[size][noise][1] for noise in noise_levels_list]
        plt.errorbar(noise_levels_list, means, yerr=stds, label=f'n={size}', capsize=5)

    plt.title("DNN Noise Profile (CIFAR-10)")
    plt.xlabel("Label Noise")
    plt.ylabel("Test Error")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wide ResNet experiment with CIFAR-10.")
    parser.add_argument("--train-path", type=str, required=True, help="Path to CIFAR-10 train set.")
    parser.add_argument("--test-path", type=str, required=True, help="Path to CIFAR-10 test set.")
    args = parser.parse_args()

    train_sizes = [5000, 10000, 30000, 50000]
    noise_levels = np.arange(0, 0.91, 0.01)
    epochs = 60
    trials = 3
    learning_rate = 0.1
    momentum = 0.9
    batch_size = 28

    experiment(train_sizes, noise_levels, epochs, trials, args.train_path, args.test_path, learning_rate, momentum,
               batch_size)