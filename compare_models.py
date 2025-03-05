import pickle as pk
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import skill_metrics as sm
from matplotlib import rcParams
import matplotlib.colors as mcolors


def compare_models():
    # Load the CSV files
    main_models_path = 'main_models/evaluation_results.csv'
    hybrid_models_path = 'hybrid_models/evaluation_results.csv'

    main_models = pd.read_csv(main_models_path)
    hybrid_models = pd.read_csv(hybrid_models_path)

    # Function to calculate percentage change
    def percentage_change(new, old):
        return ((new - old) / abs(old)) * 100

    # Merge the dataframes on test_dataset and threshold
    merged = hybrid_models.merge(main_models, on=['test_dataset', 'threshold'], suffixes=('_hybrid', '_main'))

    # Initialize lists to store results
    results = []

    # Iterate through the merged dataframe and calculate percentage changes
    for index, row in merged.iterrows():
        if row['regressor_main'] == 'RILS-ROLS':
            R2_change = percentage_change(row['R2_score_hybrid'], row['R2_score_main'])
            RMSE_change = percentage_change(row['RMSE_hybrid'], row['RMSE_main'])
            results.append({
                'regressor': row['regressor_hybrid'],
                'test_dataset': row['test_dataset'],
                'threshold': row['threshold'],
                'R2_change (%)': R2_change,
                'RMSE_change (%)': RMSE_change
            })

    # Convert the results to a dataframe
    results_df = pd.DataFrame(results)

    # Save results to a CSV file
    results_df.to_csv('hybrid_models/hybrid_models_comparison.csv', index=False)

    print("Comparison completed. Results saved to 'hybrid_models/hybrid_models_comparison.csv'")


# Define a function to extract and filter data from evaluation CSV files
def extract_evaluation_results(csv_file, dataset="Urban", threshold=4):
    df = pd.read_csv(csv_file)
    filtered_df = df[(df['test_dataset'] == dataset) & (df['threshold'] == threshold)]
    return filtered_df[['regressor', 'RMSE', 'R2_score', 'std_dev']]


# Define a function to extract data from pickle files
def extract_data_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pk.load(f)

    # Data is expected to be a list where the first element is ModelsDF
    ModelsDF = data[0] if isinstance(data, list) and len(data) > 0 else data

    # Ensure correct column names
    if 'model' not in ModelsDF.columns or 'nbytes' not in ModelsDF.columns:
        raise ValueError("Expected 'model' and 'nbytes' columns in the pickle data")

    # Only keep 'model' and 'nbytes' columns
    return ModelsDF[['model', 'nbytes']]


def generate_colors(num_colors):
    """ Generate a list of distinct colors in hexadecimal format from a custom palette. """
    custom_colors = ['#67FFF2', '#81EC86', '#B5E6BA', '#A3C4BC', '#4380AF', '#44685C', '#2A3056']
    return custom_colors[:num_colors]


def plot_model_performance(csv_file, metric='RMSE', log_x=False, log_y=False):
    # Validate metric
    if metric not in ['RMSE', 'R2_score']:
        raise ValueError("Metric must be either 'RMSE' or 'R2_score'")

    # Load data from CSV file
    df = pd.read_csv(csv_file)

    # Ensure necessary columns are present
    if not all(col in df.columns for col in ['model', 'nbytes', 'RMSE', 'R2_score']):
        raise ValueError("CSV file must contain 'model', 'nbytes', 'RMSE', and 'R2_score' columns")

    # Extract unique models
    models = df['model'].unique()

    # Define shapes and colors
    shapes = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'pentagon']
    hexagon = 'hexagon'
    num_rils_models = len([model for model in models if 'RILS-ROLS_' in model])
    rils_colors = generate_colors(num_rils_models)

    # Initialize color and shape maps
    rils_shapes = {model: hexagon for model in models if 'RILS-ROLS_' in model}
    remaining_shapes = [shape for shape in shapes if shape not in rils_shapes.values()]

    # Data for reference lines
    rils_rols_y = None
    rils_rols_x = None
    for model in models:
        if model == 'RILS-ROLS':
            model_df = df[df['model'] == model]
            rils_rols_y = model_df[metric].values[0]
            rils_rols_x = model_df['nbytes'].values[0]
            break

    # Create a blank figure
    fig = go.Figure()

    # Add horizontal and vertical lines for 'RILS-ROLS'
    if rils_rols_y is not None and rils_rols_x is not None:
        fig.add_trace(go.Scatter(
            x=[df['nbytes'].min(), df['nbytes'].max()],  # Extend line across x-axis range
            y=[rils_rols_y, rils_rols_y],  # Same y-value for the entire line
            mode='lines',
            line=dict(color='gray', dash='dash'),  # Style of the horizontal line
            name='RILS-ROLS Reference (Horizontal)',  # Label for the line
            showlegend=False  # Do not show in the legend
        ))

        fig.add_trace(go.Scatter(
            x=[rils_rols_x, rils_rols_x],  # Same x-value for the entire line
            y=[rils_rols_y, df[metric].max()],  # Start at the y-value of 'RILS-ROLS' and extend to the max y-value
            mode='lines',
            line=dict(color='gray', dash='dash'),  # Style of the vertical line
            name='RILS-ROLS Reference (Vertical)',  # Label for the line
            showlegend=False  # Do not show in the legend
        ))

    # Loop through each model and add a scatter trace
    for i, model in enumerate(models):
        model_df = df[df['model'] == model]
        if model_df.empty:
            continue

        # Assume we have one entry per model; adapt if there are multiple entries per model
        complexity = model_df['nbytes'].values[0]
        if metric == 'RMSE':
            accuracy = model_df['RMSE'].values[0]
        else:
            accuracy = model_df['R2_score'].values[0]

        # Determine marker properties
        if model == 'RILS-ROLS':
            marker_shape = 'circle'
            marker_color = 'white'
        elif 'RILS-ROLS_' in model:
            marker_shape = rils_shapes.get(model, hexagon)
            marker_color = rils_colors.pop(0)
        else:
            marker_shape = remaining_shapes[i % len(remaining_shapes)]
            marker_color = 'black'

        fig.add_trace(go.Scatter(
            x=[complexity],  # x-coordinate for the current model
            y=[accuracy],  # y-coordinate for the current model (based on selected metric)
            mode='markers',  # This explicitly states that we want our observations to be represented by points
            text=model,  # Label associated with the current model
            # Properties associated with points
            marker=dict(
                size=12,  # Size
                color=marker_color,  # Unique color for each model
                line=dict(width=1, color='black'),  # Properties of the edges
                symbol=marker_shape  # Unique shape for each model
            ),
            name=model,  # Legend label for the current model
            showlegend=True,  # Ensure the legend is displayed for each model
            legendgroup='model'  # Group by model for ordering in the legend
        ))

    # Customize the layout
    fig.update_layout(
        xaxis_title='Complexity (Bytes)',  # x-axis name
        yaxis_title=f'Error ({metric})',  # y-axis name based on selected metric
        width=700,  # Set the width of the figure to 800 pixels
        height=600,  # Set the height of the figure to 600 pixels
        plot_bgcolor='white',  # Set the background color to white
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            type='log' if log_x else 'linear'  # Set log scale if log_x is True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            type='log' if log_y else 'linear'  # Set log scale if log_y is True
        ),
        xaxis_linecolor='black',  # Set x-axis line color to black
        yaxis_linecolor='black',  # Set y-axis line color to black
        font=dict(
            family="Times New Roman",
            size=16,
            color="black"
        ),
        legend=dict(
            font=dict(size=12, color='black'),  # Set the font size and color of the legend
            title='Models'  # Title of the legend
        )
    )

    # Show the plot
    fig.show()


def evaluate_perf_compl(dataset="Urban", threshold=4, plot_metric='RMSE', log_x=False, log_y=False):
    # Paths to the pickle files
    pickle_files = ['main_models/training_results.pickle', 'hybrid_models/training_results.pickle']

    # Paths to the evaluation results CSV files
    evaluation_files = ['main_models/evaluation_results.csv', 'hybrid_models/evaluation_results.csv']

    # Create an empty DataFrame to store combined results
    combined_df = pd.DataFrame()

    # Extract and combine data from pickle files
    for pickle_file in pickle_files:
        df = extract_data_from_pickle(pickle_file)
        if not df.empty:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Extract and combine data from evaluation CSV files
    evaluation_dfs = []
    for csv_file in evaluation_files:
        df = extract_evaluation_results(csv_file, dataset=dataset, threshold=threshold)
        evaluation_dfs.append(df)

    # Concatenate all evaluation DataFrames
    evaluation_combined_df = pd.concat(evaluation_dfs, ignore_index=True)

    # Rename columns in evaluation_combined_df for merging
    evaluation_combined_df.rename(columns={'regressor': 'model'}, inplace=True)

    # Merge evaluation results into combined_df based on the model name
    # Ensure that `model` names in `combined_df` match those in `evaluation_combined_df`
    # Handle cases where there may be no matching model
    final_df = pd.merge(combined_df, evaluation_combined_df, on='model', how='left')

    # Drop any rows where essential columns might be missing
    final_df.dropna(subset=['RMSE', 'R2_score'], inplace=True)

    # Rename columns for clarity
    final_df.columns = ['model', 'nbytes', 'RMSE', 'R2_score', 'std_dev']

    # Save combined DataFrame to a CSV file
    final_df.to_csv('combined_results.csv', index=False)

    plot_model_performance('combined_results.csv', metric=plot_metric, log_x=log_x, log_y=log_y)
    # plot_taylor(csv_file='combined_results.csv')

    print('Combined results have been saved to combined_results.csv')


def plot_taylor(csv_file):
    # Set the figure properties (optional)
    rcParams["figure.figsize"] = [8.0, 6]
    rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Close any previously open graphics windows
    plt.close("all")

    # Load data from CSV file
    df = pd.read_csv(csv_file)

    # Ensure necessary columns are present
    if not all(col in df.columns for col in ['model', 'nbytes', 'RMSE', 'R2_score', 'std_dev']):
        raise ValueError("CSV file must contain 'model', 'nbytes', 'RMSE', 'R2_score', and 'std_dev' columns")

    # Extract data
    models = df['model'].unique()
    complexity = df['nbytes'].values
    rmse = df['RMSE'].values
    r = np.sqrt(df['R2_score'].values)
    sdev = np.square(df['std_dev'].values)

    #Taylor plot bug correction:
    rmse = np.append(rmse[0], rmse)
    r = np.append(r[0], r)
    sdev = np.append(sdev[0], sdev)

    # Define markers for each model
    MARKERS = {
        'RILS-ROLS': {"labelColor": "k", "symbol": "o", "size": 9, "faceColor": "w", "edgeColor": "k"},
        'RILS-ROLS': {"labelColor": "k", "symbol": "o", "size": 9, "faceColor": "w", "edgeColor": "k"},
        'SVR': {"labelColor": "k", "symbol": "s", "size": 9, "faceColor": "k", "edgeColor": "k"},
        'KNN': {"labelColor": "k", "symbol": "D", "size": 8, "faceColor": "k", "edgeColor": "k"},
        'HGB': {"labelColor": "k", "symbol": "+", "size": 11, "faceColor": "k", "edgeColor": "k"},
        'CB': {"labelColor": "k", "symbol": "^", "size": 9, "faceColor": "k", "edgeColor": "k"},
        'LRG': {"labelColor": "k", "symbol": "v", "size": 9, "faceColor": "k", "edgeColor": "k"},
        'DTR': {"labelColor": "k", "symbol": "*", "size": 12, "faceColor": "k", "edgeColor": "k"},
        'RFR': {"labelColor": "k", "symbol": "p", "size": 9, "faceColor": "k", "edgeColor": "k"},
        'RILS-ROLS_SVR': {"labelColor": "k", "symbol": "h", "size": 9, "faceColor": "c", "edgeColor": "k"},
        'RILS-ROLS_KNN': {"labelColor": "k", "symbol": "h", "size": 9, "faceColor": "w", "edgeColor": "k"},
        'RILS-ROLS_HGB': {"labelColor": "k", "symbol": "h", "size": 9, "faceColor": "m", "edgeColor": "k"},
        'RILS-ROLS_CB': {"labelColor": "k", "symbol": "h", "size": 9, "faceColor": "y", "edgeColor": "k"},
        'RILS-ROLS_LRG': {"labelColor": "k", "symbol": "h", "size": 9, "faceColor": "r", "edgeColor": "k"},
        'RILS-ROLS_DTR': {"labelColor": "k", "symbol": "h", "size": 9, "faceColor": "b", "edgeColor": "k"},
        'RILS-ROLS_RFR': {"labelColor": "k", "symbol": "h", "size": 9, "faceColor": "g", "edgeColor": "k"},
    }

    # Create a Taylor diagram
    sm.taylor_diagram(sdev, rmse, r, markers=MARKERS,
                      markerLegend='on', alpha=0.2, labelWeight='normal',
                      colOBS='k', markerobs='o', styleObs='-',
                      colRMS='#AAAAAA', styleRMS=':', widthRMS=2.0, titleRMS='on', labelRMS='RMSE', rmsLabelFormat=':.0f',
                      colsstd={'grid': '#DDDDDD', 'tick_labels': '#000000', 'ticks': '#DDDDDD', 'title': '#000000'}, styleSTD='-.', widthSTD=1.0, titleSTD='on',
                      colscor={'grid': '#DDDDDD', 'tick_labels': '#000000', 'title': '#000000'}, styleCOR='--', widthCOR=1.0, titleCOR='on', titlecorshape='linear')

    # Add annotation
    plt.annotate('Target', xy=(23000, 0), xytext=(17000, 1200),
                 arrowprops=dict(facecolor='white', arrowstyle='-'), fontsize=12)

    plt.show()
    plt.close()

