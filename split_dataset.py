import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pk
import os


def split_dataset_by_threshold():
    train_dataset = 'WHTC'
    test_datasets = ['Urban', 'Rural', 'Highway']
    thresholds = [0, 20, 40, 60, 80]

    # Create output directory if it doesn't exist
    output_dir = 'test_data'
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each combination of test_dataset and threshold
    for test_dataset in test_datasets:
        for thres in thresholds:
            # Load data from .mat file
            data1 = sio.loadmat(f'data/Dataset_PCP_{train_dataset}.mat')
            X1 = data1['X']
            y1 = data1['y']

            # Ensure y1 is a 1D array
            y1 = y1.flatten()

            # Load data from .mat file
            data2 = sio.loadmat(f'data/Dataset_PCP_{test_dataset}.mat')
            X2 = data2['X']
            y2 = data2['y']

            # Ensure y2 is a 1D array
            y2 = y2.flatten()

            # Calculate the threshold value (percentage of the maximum value of y2)
            threshold = thres / 100 * np.max(y2)

            # Create a boolean mask where y2 is higher than the threshold
            mask = y2 > threshold

            # Use the mask to select the rows from X2 and corresponding elements from y2
            X2 = X2[mask]
            y2 = y2[mask]

            # Concatenate the datasets
            X = np.concatenate((X1, X2), axis=0)
            y = np.concatenate((y1, y2), axis=0).ravel()

            np.random.seed(31415)

            # Training parameters
            train_size = X1.shape[0] / X.shape[0]  # train size

            # Split data, train and evaluate model
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0, shuffle=False)

            # Create a file name based on the current combination of test_dataset and threshold
            file_name = f'{output_dir}/Dataset_PCP_{test_dataset}_thres_{thres}.pickle'

            # Save the data to a pickle file
            with open(file_name, "wb") as fname:
                pk.dump((X_train, X_test, y_train, y_test), fname)

            print(f'Saved data for test_dataset={test_dataset}, threshold={thres} to {file_name}')


def split_dataset_by_quartiles():
    train_dataset = 'WHTC'
    test_datasets = ['WHTC', 'Urban', 'Rural', 'Highway']

    # Create output directory if it doesn't exist
    output_dir = 'test_data'
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each test dataset
    for test_dataset in test_datasets:
        # Load data from .mat file
        data1 = sio.loadmat(f'data/Dataset_PCP_{train_dataset}.mat')
        X1 = data1['X']
        y1 = data1['y']

        # Ensure y1 is a 1D array
        y1 = y1.flatten()

        # Load data from .mat file
        data2 = sio.loadmat(f'data/Dataset_PCP_{test_dataset}.mat')
        X2 = data2['X']
        y2 = data2['y']

        # Ensure y2 is a 1D array
        y2 = y2.flatten()

        # Sort X2 and y2 based on y2 values
        sorted_indices = np.argsort(y2)
        X2_sorted = X2[sorted_indices]
        y2_sorted = y2[sorted_indices]

        # Determine the quartile indices based on the number of samples
        quartile_size = len(y2_sorted) // 4
        quartile_indices = [quartile_size, 2 * quartile_size, 3 * quartile_size]

        # Split data into quartiles
        # for i, (start_idx, end_idx) in enumerate(zip([0] + quartile_indices, quartile_indices + [len(y2_sorted)])):
        for i, (start_idx, end_idx) in enumerate(zip([0] + quartile_indices, [len(y2_sorted), len(y2_sorted), len(y2_sorted), len(y2_sorted)])):
            X2_quartile = X2_sorted[start_idx:end_idx]
            y2_quartile = y2_sorted[start_idx:end_idx]

            # Concatenate the datasets
            X = np.concatenate((X1, X2_quartile), axis=0)
            y = np.concatenate((y1, y2_quartile), axis=0).ravel()

            np.random.seed(31415)

            # Training parameters
            train_size = X1.shape[0] / X.shape[0]  # train size

            # Split data, train and evaluate model
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0, shuffle=False)

            # Create a file name based on the current combination of test_dataset and quartile
            file_name = f'{output_dir}/Dataset_PCP_{test_dataset}_quartile_{i + 1}.pickle'

            # Save the data to a pickle file
            with open(file_name, "wb") as fname:
                pk.dump((X_train, X_test, y_train, y_test), fname)

            print(f'Saved data for test_dataset={test_dataset}, quartile={i + 1} to {file_name}')