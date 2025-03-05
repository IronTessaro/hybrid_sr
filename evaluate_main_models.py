from model_train_lib import evaluate_results


def evaluate_main_models():
    # Specify the file name containing the ModelsDF
    models_file_name = 'main_models/training_results.pickle'
    output_dir = 'main_models'

    evaluate_results(input_file=models_file_name, output_dir=output_dir)
