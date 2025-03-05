import pickle as pk
from sklearn.model_selection import train_test_split
from model_train_lib import train_models


def train_residuals_models(dataset='main_models/residuals_data.pickle', kfolds=2, repeats=1, n_search=1, score='r2'):
    # Specify the path to your residuals pickle file
    main_model = 'RILS-ROLS'

    # Load the data from the pickle file
    with open(dataset, 'rb') as file:
        data = pk.load(file)

    if 'augmented' in dataset:
        X_res = data[0]
        y_res = data[1]
    else:
        # Extract X_train and residuals
        PredDF_loaded = data[0]
        modelName_loaded = data[1]
        X_res = data[2]

        # Extract the residuals from PredDF
        y_res = PredDF_loaded[PredDF_loaded['model'] == main_model]['residuals']

    print("Loaded X_res and y_res from the pickle file.")

    train_size = 0.5
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size=train_size, random_state=0, shuffle=True)

    print(X_train.shape)
    print(y_train.shape)

    print(X_test.shape)
    print(y_test.shape)

    # ---------- Training ------------:

    train_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, output_dir='residuals',
                 kfolds=kfolds, repeats=repeats, n_search=n_search, score=score)

