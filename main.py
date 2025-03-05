from time import time
from split_dataset import split_dataset_by_threshold, split_dataset_by_quartiles
from train_main import train_main_models
from evaluate_main_models import evaluate_main_models
from train_residuals import train_residuals_models
from evaluate_residuals_models import evaluate_residuals_models
from build_hybrid_models import build_hybrid_models
from evaluate_hybrid_models import evaluate_hybrid_models
from compare_models import compare_models, evaluate_perf_compl
from model_train_lib import *


def main():
    start_time = time()

    split_dataset_by_threshold()
    split_dataset_by_quartiles()
    augment_data(dataset='data/Dataset_PCP_WHTC.mat', num_bins=31, target_count=4000)
    train_main_models(dataset='data/Dataset_PCP_WHTC.mat', kfolds=2, repeats=1, n_search=1000, score='r2')
    evaluate_main_models()
    augment_data(dataset='main_models/residuals_data.pickle', num_bins=20, target_count=9000)
    train_residuals_models(dataset='main_models/residuals_data.pickle', kfolds=2, repeats=1, n_search=1000, score='r2')
    evaluate_residuals_models()
    build_hybrid_models(main_regressor_name="RILS-ROLS")
    evaluate_hybrid_models()
    compare_models()
    evaluate_perf_compl(dataset="Urban", threshold=4, plot_metric='RMSE', log_x=True, log_y=True)

    print(f"\nTotal execution time: {time() - start_time} seconds")


if __name__ == '__main__':
    main()
