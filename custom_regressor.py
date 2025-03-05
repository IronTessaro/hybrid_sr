# custom_regressor.py
import numpy as np


class CustomRegressorSR:
    def __init__(self, expression):
        self.expression = expression

    def predict(self, X):
        x_dict = {f'x{i}': X[:, i] for i in range(X.shape[1])}

        # Add all necessary functions from numpy to eval context
        eval_context = {name: getattr(np, name) for name in dir(np) if callable(getattr(np, name))}

        y = eval(self.expression, eval_context, x_dict)

        return y


# Custom regressor class
class CustomRegressorHybrid:
    def __init__(self, main_regressor, residual_regressor):
        self.main_regressor = main_regressor
        self.residual_regressor = residual_regressor

    def predict(self, X):
        main_pred = self.main_regressor.predict(X)
        residual_pred = self.residual_regressor.predict(X)
        return main_pred + residual_pred
