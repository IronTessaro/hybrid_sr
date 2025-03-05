from model_train_lib import evaluate_results


def evaluate_hybrid_models():
    # Specify the file name containing the ModelsDF
    models_file_name = 'hybrid_models/training_results.pickle'
    output_dir = 'hybrid_models'

    evaluate_results(input_file=models_file_name, output_dir=output_dir)
