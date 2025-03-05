import pickle as pk
import os
import pandas as pd
from custom_regressor import CustomRegressorHybrid


def build_hybrid_models(main_regressor_name="RILS-ROLS"):
    # Load the main and residual models from the pickle files
    with open('main_models/training_results.pickle', 'rb') as f:
        main_data = pk.load(f)
        ModelsDF_main, modelName_main = main_data

    with open('residuals/training_results.pickle', 'rb') as f:
        residual_data = pk.load(f)
        ModelsDF_residual, modelName_residual = residual_data

    main_regressor = ModelsDF_main.loc[ModelsDF_main['model'] == main_regressor_name, 'regressor'].values[0]
    main_regressor_nbytes = ModelsDF_main.loc[ModelsDF_main['model'] == main_regressor_name, 'nbytes'].values[0]

    # Initialize custom regressors with "RILS-ROLS" as the main regressor and each of the other regressors as residual regressors
    custom_regressors = []
    custom_model_names = []
    custom_nbytes = []

    for i, row in ModelsDF_residual.iterrows():
        residual_regressor = row['regressor']
        residual_model_name = row['model']
        residual_regressor_nbytes = row['nbytes']
        total_nbytes = main_regressor_nbytes + residual_regressor_nbytes

        custom_regressor = CustomRegressorHybrid(main_regressor, residual_regressor)
        custom_regressors.append(custom_regressor)
        custom_model_names.append(f"{main_regressor_name}_{residual_model_name}")
        custom_nbytes.append(total_nbytes)

    # Create a DataFrame to save the custom regressors
    ModelsDF = pd.DataFrame({
        "model": custom_model_names,
        "regressor": custom_regressors,
        "nbytes": custom_nbytes  # Adjust this if you want to save the size
    })

    # Save the custom regressors in the specified format
    output_dir = 'hybrid_models'
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.join(output_dir, f'training_results.pickle')
    with open(file_name, "wb") as fname:
        pk.dump([ModelsDF, custom_model_names], fname)
    print(f'File saved as: {file_name}')
