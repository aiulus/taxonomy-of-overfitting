import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def generate_data(d, N):
    u = np.zeros(d)
    v = np.identity(d)
    points = np.random.multivariate_normal(mean=u, cov=v, size=N)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    S_d = points / norms
    return S_d


def gaussian_kernel(X, Y, sigma=1.0):
    pairwise_sq_dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
    return np.exp(-pairwise_sq_dists / (2 * sigma ** 2))


def kernel_ridge_regression(X_train, y_train, X_test, ridge=0.1, sigma=1.0):
    K = gaussian_kernel(X_train, X_train, sigma) + ridge * np.eye(X_train.shape[0])
    alpha = np.linalg.solve(K, y_train)
    K_test = gaussian_kernel(X_test, X_train, sigma)
    y_pred = K_test @ alpha
    return y_pred


def experiment(d, sample_sizes, num_runs=100):
    mse_results = {size: [] for size in sample_sizes}

    for _ in tqdm(range(num_runs), desc=f"Dimension {d}"):
        for size in sample_sizes:
            X_train = generate_data(d, size)
            y_train = np.random.normal(0, 1, size)
            X_test = generate_data(d, 100)  # Test set size fixed to 100
            y_test = np.zeros(100)  # Clean labels

            y_pred = kernel_ridge_regression(X_train, y_train, X_test, ridge=0.1, sigma=1.0)
            mse = mean_squared_error(y_test, y_pred)
            mse_results[size].append(mse)

    return mse_results


def plot_results(results, sample_sizes, dimensions):
    plt.figure(figsize=(10, 6))

    for d, mse_results in results.items():
        means = [np.mean(mse_results[size]) for size in sample_sizes]
        quantiles_25 = [np.quantile(mse_results[size], 0.25) for size in sample_sizes]
        quantiles_75 = [np.quantile(mse_results[size], 0.75) for size in sample_sizes]

        plt.plot(sample_sizes, means, label=f'd={d}')
        plt.fill_between(sample_sizes, quantiles_25, quantiles_75, alpha=0.2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Sizes')
    plt.ylabel('Test MSE')
    plt.title('Ridged Gaussian Kernel')
    plt.xticks([10, 100, 1000, 10000], labels=['10', '100', '1000', '10000'])
    plt.yticks([1, 0.1, 0.01, 0.001], labels=['$10^0$', '$10^{-1}$', '$10^{-2}$', '$10^{-3}$'])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sample_sizes = np.logspace(0.7, 4, num=50, dtype=int)
    dimensions = [5, 10, 15]

    results = {}
    for d in dimensions:
        results[d] = experiment(d, sample_sizes, num_runs=100)

    plot_results(results, sample_sizes, dimensions)
