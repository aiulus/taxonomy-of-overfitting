import os
import argparse

import numpy as np
import pandas as pd

import torch
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, Lambda

def generate_data(d, N):
    """
        Generate N uniformly distributed points on the surface of a (d-1)-dimensional unit sphere.

        Parameters:
        d (int): Dimension of the space (d-1 dimensional sphere).
        N (int): Number of points to generate.

        Returns:
        np.ndarray: An array of shape (N, d) containing the Cartesian coordinates of the points.
    """
    # Generate d-dimensional zero vector
    u = np.zeros(d)
    # Generate dxd - dimensional identity matrix
    v = np.identity(d)

    np.random.seed(42)

    # Generate N points from a d-dimensional standard normal distribution
    points = np.random.multivariate_normal(mean=u, cov=v, size=N)

    # Normalize each point to lie on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    S_d = points / norms

    return S_d

def make_csv(data, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the full path
    filepath = os.path.join(directory, filename)

    # Save the points to a CSV file
    header = ','.join([f'x{i+1}' for i in range(data.shape[1])])
    np.savetxt(filepath, data, delimiter=",", header=header, comments="")
    print(f"Points saved to {filepath}")


def process_mnist():
    transform = Lambda(lambda x: x.view(-1).numpy())

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

    # Extract the data and targets
    train_data = train_dataset.data.view(train_dataset.data.size(0), -1).numpy()
    train_targets = train_dataset.targets.numpy()
    test_data = test_dataset.data.view(test_dataset.data.size(0), -1).numpy()
    test_targets = test_dataset.targets.numpy()

    # Save the flattened training and test sets to .csv files
    train_df = pd.DataFrame(train_data)
    train_df['label'] = train_targets
    train_df.to_csv('mnist_train.csv', index=False)

    test_df = pd.DataFrame(test_data)
    test_df['label'] = test_targets
    test_df.to_csv('mnist_test.csv', index=False)

    # Map the original targets to binary labels
    train_binary_labels = [1 if target % 2 == 0 else -1 for target in train_targets]
    test_binary_labels = [1 if target % 2 == 0 else -1 for target in test_targets]

    # Save the binary labeled datasets
    train_df['binary_label'] = train_binary_labels
    train_df.to_csv('binary_mnist_train.csv', index=False)

    test_df['binary_label'] = test_binary_labels
    test_df.to_csv('binary_mnist_test.csv', index=False)

    print("MNIST files saved successfully.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate data on the surface of a unit sphere and save it to a CSV file.")
    parser.add_argument("-d", "--dimension", type=int, required=True, help="Dimension of the space (d-1 dimensional sphere).")
    parser.add_argument("-N", "--points", type=int, required=True, help="Number of points to generate.")
    parser.add_argument("-o", "--output-dir", type=str, default="data\synthetic",
                        help="Output directory for saving the CSV file.")
    #parser.add_argument("-f", "--filename", type=str, default="points_on_sphere.csv", help="Filename for the CSV file.")

    args = parser.parse_args()

    # Generate data
    data = generate_data(args.dimension, args.points)

    # Construct filename with specific dimensionality and sample size
    filename = f"synt_d{args.dimension}_n{args.points}"

    # Save data to CSV
    make_csv(data, args.output_dir, filename)
