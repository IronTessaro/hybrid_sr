# custom_regressor.py
import os
import numpy as np
import pickle as pk
import glob
import sys
import pandas as pd
from tqdm import tqdm  # progress bar no for loop
from time import time
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
# from scipy.stats import randint, loguniform, reciprocal
from scipy import stats
import scipy.io as sio
import sympy as sp
import re
# from scipy.stats.stats import mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    RandomizedSearchCV,
)
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from rils_rols.rils_rols import RILSROLSRegressor
from rils_rols.rils_rols_ensemble import RILSROLSEnsembleRegressor
from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor as BINGORegressor
# from pyoperon.sklearn import SymbolicRegressor as OPERONRegressor
from itea.regression import ITEA_regressor
from powershap import PowerShap
import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution, LogUniformDistribution
from concurrent.futures import ThreadPoolExecutor, as_completed
from custom_regressor import CustomRegressorSR


def augment_data(dataset, num_bins, target_count):
    if dataset.endswith(".mat"):
        # Load data from .mat file
        data1 = sio.loadmat(dataset)
        X = data1['X']
        y = data1['y'].flatten()
    elif dataset.endswith(".pickle"):
        main_model = 'RILS-ROLS'

        # Load the data from the pickle file
        with open(dataset, 'rb') as file:
            data = pk.load(file)

        # Extract X_train and residuals
        PredDF_loaded = data[0]
        modelName_loaded = data[1]
        X = data[2]

        # Extract the residuals from PredDF
        y = PredDF_loaded[PredDF_loaded['model'] == main_model]['residuals'].to_numpy().flatten()
    else:
        print("Invalid dataset file name!")

    # Step 1: Create bin edges dynamically
    bins = np.linspace(y.min(), y.max(), num_bins + 1)

    # Step 2: Bin y
    bin_indices = np.digitize(y, bins=bins, right=False)

    # Containers for augmented data
    X_augmented = []
    y_augmented = []

    # Step 3: Augment data for each bin
    for bin_num in np.unique(bin_indices):
        bin_mask = bin_indices == bin_num
        X_bin = X[bin_mask]
        y_bin = y[bin_mask]

        # Calculate how many more samples are needed
        num_samples_needed = max(0, target_count - len(y_bin))  # Ensure non-negative values

        if num_samples_needed > 0:
            # Randomly choose indices to replicate
            replicate_indices = np.random.choice(len(y_bin), num_samples_needed, replace=True)
            X_bin_augmented = np.vstack([X_bin, X_bin[replicate_indices]])
            y_bin_augmented = np.concatenate([y_bin, y_bin[replicate_indices]])
        else:
            X_bin_augmented = X_bin
            y_bin_augmented = y_bin

        # Limit the number of samples per bin to exactly target_count
        if len(y_bin_augmented) > target_count:
            X_bin_augmented = X_bin_augmented[:target_count]
            y_bin_augmented = y_bin_augmented[:target_count]

        X_augmented.append(X_bin_augmented)
        y_augmented.append(y_bin_augmented)

    # Combine all augmented bins together
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.concatenate(y_augmented)

    # Save augmented data to a new .mat file
    save_augmented_data(dataset, X_augmented, y_augmented)

    # Plot the histogram before and after augmentation
    plot_histogram(y, y_augmented, num_bins)

    return X_augmented, y_augmented


def save_augmented_data(original_dataset, X_augmented, y_augmented):
    if original_dataset.endswith(".mat"):
        new_filename = original_dataset.replace('.mat', '_augmented.mat')
        sio.savemat(new_filename, {'X': X_augmented, 'y': y_augmented})

    elif original_dataset.endswith(".pickle"):
        new_filename = original_dataset.replace('.pickle', '_augmented.pickle')
        with open(new_filename, "wb") as fname:
            pk.dump([X_augmented, y_augmented], fname)

    print(f"Augmented data saved as: {new_filename}")


def plot_histogram(y_before, y_after, num_bins):
    plt.figure(figsize=(10, 6))

    # Plot the histogram before augmentation
    plt.hist(y_before, bins=num_bins, color='#4380AF', edgecolor='black', alpha=0.6, label='Before Augmentation')

    # Plot the histogram after augmentation
    plt.hist(y_after, bins=num_bins, color='#D62728', edgecolor='black', alpha=0.6, label='After Augmentation')

    plt.xlabel('Target Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Target Values Before and After Augmentation')
    plt.legend(loc='upper right')
    plt.grid(True, color='lightgray')
    plt.show()


def build_mlp(hidden_layer_1, hidden_layer_2, **kwargs):
    # Combine the two hidden layer sizes into a tuple and pass it to MLPRegressor
    return MLPRegressor(hidden_layer_sizes=(hidden_layer_1, hidden_layer_2), early_stopping=True, max_iter=1000, **kwargs)


def train_models(X_train, y_train, X_test, y_test, output_dir, kfolds=2, repeats=1, n_search=1, score='r2'):

    # List of hyperparameters to be tested within the hyperparameter random search
    param_grids = []
    # Define models
    models = []

    # ------------------------------------------------------
    tfuncs = {
        'log': np.log,
        'sqrt.abs': lambda x: np.sqrt(np.abs(x)),
        'id': lambda x: x,
        'sin': np.sin,
        'cos': np.cos,
        'exp': np.exp
    }

    tfuncs_dx = {
        'log': lambda x: 1 / x,
        'sqrt.abs': lambda x: x / (2 * (np.abs(x) ** (3 / 2))),
        'id': lambda x: np.ones_like(x),
        'sin': np.cos,
        'cos': lambda x: -np.sin(x),
        'exp': np.exp,
    }

    # ------------------------------------------------------
    # Adding GaussianProcessRegressor
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    # param_grids.append(  # GPR
    #     {
    #         "GPR__alpha": FloatDistribution(1e-10, 1e-1, log=True),
    #         "GPR__kernel": CategoricalDistribution([kernel]),
    #         "GPR__n_restarts_optimizer": IntDistribution(0, 10)
    #     }
    # )
    # models.append(("GPR", GaussianProcessRegressor()))
    # ------------------------------------------------------

    # param_grids.append(  # ITEA
    #     {
    #         "ITEA__gens": IntDistribution(5, 10),
    #         "ITEA__popsize": IntDistribution(30, 50),
    #         "ITEA__max_terms": IntDistribution(10, 20),
    #         "ITEA__expolim": CategoricalDistribution([(0, 2)]),
    #         "ITEA__verbose": CategoricalDistribution([False]),
    #         "ITEA__tfuncs": CategoricalDistribution([tfuncs]),
    #         "ITEA__tfuncs_dx": CategoricalDistribution([tfuncs_dx]),
    #         "ITEA__random_state": IntDistribution(42, 42),
    #         "ITEA__simplify_method": CategoricalDistribution(['simplify_by_coef'])
    #     }
    # )
    # models.append(("ITEA", ITEA_regressor()))

    # # ------------------------------------------------------
    # param_grids.append(  # OPERON
    #     {
    #         "OPERON__allowed_symbols": CategoricalDistribution(["add,sub,mul,div,constant,variable"]),
    #         "OPERON__brood_size": IntDistribution(5, 10),
    #         "OPERON__comparison_factor": FloatDistribution(0.0, 0.0),
    #         "OPERON__crossover_internal_probability": FloatDistribution(0.5, 0.9),
    #         "OPERON__crossover_probability": FloatDistribution(0.5, 1.0),
    #         "OPERON__epsilon": FloatDistribution(1e-06, 1e-05, log=True),
    #         "OPERON__female_selector": CategoricalDistribution(["tournament"]),
    #         "OPERON__generations": IntDistribution(300, 500),
    #         "OPERON__initialization_max_depth": IntDistribution(2, 5),
    #         "OPERON__initialization_max_length": IntDistribution(5, 10),
    #         "OPERON__initialization_method": CategoricalDistribution(["btc"]),
    #         "OPERON__irregularity_bias": FloatDistribution(0.0, 0.0),
    #         "OPERON__male_selector": CategoricalDistribution(["tournament"]),
    #         "OPERON__max_depth": IntDistribution(10, 20),
    #         "OPERON__max_evaluations": IntDistribution(10000, 10000),
    #         "OPERON__max_length": IntDistribution(10, 20),
    #         "OPERON__max_selection_pressure": IntDistribution(50, 100),
    #         "OPERON__mutation_probability": FloatDistribution(0.1, 0.5),
    #         "OPERON__n_threads": IntDistribution(32, 32),
    #         "OPERON__offspring_generator": CategoricalDistribution(["basic"]),
    #         "OPERON__pool_size": IntDistribution(500, 1000),
    #         "OPERON__population_size": IntDistribution(200, 500),
    #         "OPERON__random_state": CategoricalDistribution([None]),
    #         "OPERON__reinserter": CategoricalDistribution(["keep-best"]),
    #         "OPERON__time_limit": IntDistribution(300, 300),
    #         "OPERON__tournament_size": IntDistribution(1, 3)
    #     }
    # )
    # models.append(("OPERON", OPERONRegressor()))

    # ------------------------------------------------------
    # param_grids.append(  # BINGO
    #     {
    #         "BINGO__population_size": IntDistribution(30, 50),
    #         "BINGO__stack_size": IntDistribution(5, 10),
    #         "BINGO__use_simplification": CategoricalDistribution([True]),
    #         "BINGO__max_time": IntDistribution(100, 100)
    #     }
    # )
    # models.append(("BINGO", BINGORegressor()))

    # ------------------------------------------------------
    # param_grids.append(  # RILS-ROLS
    #     {
    #         "RILS-ROLS__sample_size": IntDistribution(1, 1),
    #         "RILS-ROLS__complexity_penalty": FloatDistribution(1e-3, 0.5, log=True),
    #         "RILS-ROLS__max_complexity": IntDistribution(20, 300),
    #     }
    # )
    # models.append(("RILS-ROLS", RILSROLSRegressor(max_seconds=30)))

    # ------------------------------------------------------
    param_grids.append(  # SVR
        {

            "SVR__C": FloatDistribution(1, 1e6, log=True),
            "SVR__epsilon": FloatDistribution(1e-6, 1, log=True),
            "SVR__kernel": CategoricalDistribution(['linear', 'rbf']),
        }
    )
    models.append(("SVR", SVR(gamma='auto', max_iter=10000)))

    # ------------------------------------------------------
    param_grids.append(  # MLP
        {
            "MLP__hidden_layer_sizes": IntDistribution(100, 1000),
            "MLP__alpha": FloatDistribution(1e-6, 1, log=True),
            "MLP__learning_rate": CategoricalDistribution(['constant', 'invscaling', 'adaptive']),
            "MLP__validation_fraction": FloatDistribution(0.1, 0.8),
        }
    )

    # Create the MLP model and append to models
    models.append(("MLP", MLPRegressor(early_stopping=True, max_iter=1000)))

    # ------------------------------------------------------
    param_grids.append(  # kNN
        {
            "KNN__n_neighbors": IntDistribution(5, 20),
            "KNN__p": IntDistribution(1, 2),
        }
    )
    models.append(('KNN', KNeighborsRegressor(weights="distance", n_jobs=6)))

    # ------------------------------------------------------
    param_grids.append(  # Histogram Gradient Boost
        {
            "HGB__learning_rate": FloatDistribution(0.001, 0.1, log=True),
            "HGB__max_iter": IntDistribution(5, 100),
            "HGB__max_leaf_nodes": IntDistribution(5, 25),
            "HGB__max_depth": IntDistribution(2, 7),
            "HGB__min_samples_leaf": IntDistribution(20, 60),
            "HGB__l2_regularization": FloatDistribution(0.01, 1, log=True),
            "HGB__max_features": FloatDistribution(0.5, 0.8),
            "HGB__validation_fraction": FloatDistribution(0.05, 0.2),
            "HGB__tol": FloatDistribution(0.0001, 0.001, log=True),
        }
    )
    models.append(("HGB", HistGradientBoostingRegressor(early_stopping=True)))

    # ------------------------------------------------------
    param_grids.append(  # CatBoost
        {
            "CB__iterations": IntDistribution(5, 100),
            "CB__learning_rate": FloatDistribution(0.001, 0.1, log=True),
            "CB__depth": IntDistribution(3, 7),
            "CB__l2_leaf_reg": IntDistribution(5, 15),
            "CB__model_size_reg": FloatDistribution(10, 10000, log=True),
            "CB__rsm": FloatDistribution(0.3, 0.8),
            "CB__bagging_temperature": IntDistribution(1, 4),
            "CB__subsample": FloatDistribution(0.1, 0.8),
            "CB__random_strength": IntDistribution(1, 5),
            "CB__min_data_in_leaf": IntDistribution(10, 50),
            "CB__max_leaves": IntDistribution(10, 30),
            "CB__model_shrink_rate": FloatDistribution(0.001, 0.1, log=True),
        }
    )
    models.append(("CB", CatBoostRegressor(grow_policy="Lossguide", od_type="Iter", verbose=False)))

    # ------------------------------------------------------
    param_grids.append(  # LRG
        {
            "LRG__alpha": FloatDistribution(1e-2, 1e2, log=True)
        }
    )
    models.append(("LRG", Ridge()))

    # ------------------------------------------------------
    param_grids.append(  # DTR
        {
            "DTR__max_depth": IntDistribution(2, 20),
            "DTR__min_samples_split": IntDistribution(2, 10),
            "DTR__min_samples_leaf": IntDistribution(3, 10),
            "DTR__max_leaf_nodes": IntDistribution(1, 50),
            "DTR__ccp_alpha": FloatDistribution(0.01, 0.1),
            "DTR__max_features": FloatDistribution(0.0, 1.0)
        }
    )
    models.append(('DTR', DecisionTreeRegressor()))

    # ------------------------------------------------------
    param_grids.append(  # RFR
        {
            "RFR__criterion": CategoricalDistribution(['squared_error', 'friedman_mse']),
            "RFR__max_depth": IntDistribution(1, 10),
            "RFR__min_samples_split": IntDistribution(3, 10),
            "RFR__min_samples_leaf": IntDistribution(3, 10),
            "RFR__max_features": FloatDistribution(0.0, 0.1),
            "RFR__max_leaf_nodes": IntDistribution(1, 50),
            "RFR__ccp_alpha": FloatDistribution(0.01, 0.1)
        }
    )
    models.append(("RFR", RandomForestRegressor(n_estimators=100, bootstrap=True)))

    # ------------------------------------------------------

    # Prepare your model names and counts
    modelName = [tup[0] for tup in models]
    Nmodels = len(modelName)

    # Store results
    PredDF = pd.DataFrame(columns=["y", "yh", "model"])
    ModelsDF = pd.DataFrame(columns=["model", "nbytes"])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    init_time = time()

    # procedure beginning
    # for k in tqdm(range(Nmodels)):
    for k in range(Nmodels):
        print("")
        print("Model: %s " % modelName[k])

        rkf = RepeatedKFold(n_splits=kfolds, n_repeats=repeats, random_state=0)
        # clf = Pipeline(steps=[("pca", PCA(n_components=0.95)), ("scaler", StandardScaler()), (models[k])])
        # clf = Pipeline(steps=[("scaler", StandardScaler()), (models[k])])
        clf = Pipeline(steps=[(models[k])])
        param_grid = param_grids[k]

        # random_search = RandomizedSearchCV(clf,verbose=10,scoring="neg_mean_squared_error",
        #                                  param_distributions=param_grid,n_iter=n_search,n_jobs=-1,
        #                                  cv=rkf,random_state=0)

        random_search = optuna.integration.OptunaSearchCV(clf, verbose=200, scoring=score,
                                                          param_distributions=param_grid, n_trials=n_search, n_jobs=1,
                                                          refit=True, random_state=0, timeout=None)

        # selector = PowerShap(model=models[k],
        #                           param_grid=param_grid,
        #                           n_iter=n_search,
        #                           cv=rkf)
        # selector.fit(X, y)  # Fit the PowerShap feature selector
        # selector.transform(X)  # Reduce the dataset to the selected features

        start_time = time()
        random_search.fit(X_train, y_train)
        end_time = time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        yh_test = random_search.predict(X_test).ravel()

        if modelName[k] == "RILS-ROLS":
            best_pipeline = random_search.best_estimator_
            rils_rols_regressor = best_pipeline.named_steps['RILS-ROLS']
            parsed_expr = sp.sympify(rils_rols_regressor.model_string())
            expression_str = str(parsed_expr)
            regressor = CustomRegressorSR(expression_str)
            expression_bytes = expression_str.encode('utf-8')
            nbytes = len(expression_bytes)

        elif modelName[k] == "ITEA":
            best_pipeline = random_search.best_estimator_
            itea_regressor = best_pipeline.named_steps['ITEA']
            parsed_expr = itea_regressor.bestsol_
            expression_str = str(parsed_expr)
            regressor = CustomRegressorSR(expression_str)
            expression_bytes = expression_str.encode('utf-8')
            nbytes = len(expression_bytes)

        elif modelName[k] == "BINGO":
            best_pipeline = random_search.best_estimator_
            bingo_regressor = best_pipeline.named_steps['BINGO']
            parsed_expr = bingo_regressor.get_best_individual()
            expression_str = str(parsed_expr)
            regressor = CustomRegressorSR(expression_str)
            expression_bytes = expression_str.encode('utf-8')
            nbytes = len(expression_bytes)

        else:
            # ------------- save data for analysis
            with open('temp.pickle', 'wb') as handle:  # number of bytes in the classifier
                pk.dump(random_search.best_estimator_, handle, protocol=pk.HIGHEST_PROTOCOL)
            # p = pk.dumps(random_search.best_estimator_, open('model123.h5', 'wb'))
            nbytes = os.stat('temp.pickle').st_size
            regressor = random_search.best_estimator_

        yh_train = random_search.predict(X_train).ravel()
        residuals = y_train - yh_train

        PredDF = pd.concat([PredDF,
                            pd.concat([
                                pd.DataFrame(residuals, columns=["residuals"]),
                                pd.DataFrame(
                                    np.repeat(modelName[k], yh_test.shape), columns=["model"])],
                                axis=1,
                            )], ignore_index=True)

        ModelsDF = pd.concat([ModelsDF,
                              pd.concat([
                                  pd.DataFrame([modelName[k]], columns=["model"]),
                                  pd.DataFrame([regressor], columns=["regressor"]),
                                  pd.DataFrame([nbytes], columns=["nbytes"]),
                              ], axis=1,
                              )], ignore_index=True)

    now = datetime.now()
    formatted_date_time = now.strftime("%Y%m%d_%H%M%S")

    # file_name = f'{output_dir}/{formatted_date_time}_k{kfolds}r{nk}s{n_search}_training.pickle'
    file_name = os.path.join(output_dir, 'training_results.pickle')
    with open(file_name, "wb") as fname:
        pk.dump([ModelsDF, modelName], fname)
    print(f'File saved as: {file_name}')

    # file_name = f'{output_dir}/{formatted_date_time}_k{kfolds}r{nk}s{n_search}_residuals.pickle'
    file_name = os.path.join(output_dir, 'residuals_data.pickle')
    with open(file_name, "wb") as fname:
        pk.dump([PredDF, modelName, X_train], fname)
    print(f'File saved as: {file_name}')

    end_time = time()
    execution_time = end_time - init_time
    print(f"\nTotal execution time: {execution_time} seconds")


def evaluate_results(input_file, output_dir):
    # Path to the directory containing the pickle files
    test_dir = 'test_data'
    file_pattern = os.path.join(test_dir, '*.pickle')

    # Open the file and load the data
    with open(input_file, "rb") as fname:
        data = pk.load(fname)

    # Unpack the data
    ModelsDF, modelName = data

    # Initialize a list to store the results and predictions
    results_list = []
    predictions_list = []

    # Loop over all regressors in ModelsDF
    for index, row in ModelsDF.iterrows():
        regressor = row["regressor"]
        regressor_name = row["model"]

        # Loop over all the pickle files in the output directory
        for file_name in glob.glob(file_pattern):
            # Extract test_dataset and threshold from the file name
            base_name = os.path.basename(file_name)
            parts = base_name.split('_')
            test_dataset = parts[2]
            threshold = parts[4].split('.')[0]

            # Open the file and load the data
            with open(file_name, "rb") as fname:
                X_train, X_test, y_train, y_test = pk.load(fname)

            # Use the regressor to make predictions
            y_pred = regressor.predict(X_test)

            # Calculate the R² score
            r2 = r2_score(y_test, y_pred)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Calculate Standard Deviation of Predictions
            std_dev = np.std(y_pred)

            # Append the results to the list
            results_list.append({
                'regressor': regressor_name,
                'test_dataset': test_dataset,
                'threshold': threshold,
                'R2_score': r2,
                'RMSE': rmse,
                'std_dev': std_dev
            })

            # Append the predictions to the predictions list
            predictions_list.append({
                'regressor': regressor_name,
                'test_dataset': test_dataset,
                'threshold': threshold,
                'y_test': y_test,
                'y_pred': y_pred
            })

            print(
                f'Evaluated R² for regressor={regressor_name}, test_dataset={test_dataset}, threshold={threshold}: R² = {r2}, RMSE = {rmse}, Std Dev = {std_dev}')

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)

    # Save the predictions to a pickle file
    predictions_file_name = os.path.join(output_dir, 'predictions.pickle')
    with open(predictions_file_name, "wb") as fname:
        pk.dump(predictions_list, fname)

    print(f'Predictions saved to: {predictions_file_name}')
