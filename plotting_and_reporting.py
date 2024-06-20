from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from palettable.colorbrewer.qualitative import Set1_9, Pastel2_3
from matplotlib.colors import ListedColormap


def write_evaluation_results(evaluation_results, eval_name="summary"):
    """
    Writes evaluation results to a text file.

    :param evaluation_results: The evaluation results to write to the file.
    :param eval_name: Optional. The name of the evaluation. Defaults to "summary".
    :return: None
    """
    # Get current date and time
    now = datetime.now()
    # Format date and time as a string in the format "ddmmyyyyHHMMSS"
    dt_string = now.strftime("%d%m%Y%H%M%S")
    # Construct output file name using formatted date-time and evaluation name
    output_file_name = f"reports/{eval_name}-evaluation_results-{dt_string}.txt"
    # Write evaluation results to the output file
    with open(output_file_name, "w") as f:
        f.write(evaluation_results)


def plot_model_comparisons(results_df):
    """
    Plots comparisons between different models based on specified metrics.

    :param results_df: A pandas DataFrame containing the results of model comparisons.
    :return: None
    """
    # Check if input is a pandas DataFrame
    if not isinstance(results_df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame.")

    # Define required columns for comparison
    required_columns = {'Model', 'Total Cost(1000)', 'Time Length', 'Summary Score'}
    # Check if required columns are present in the DataFrame
    if not required_columns.issubset(results_df.columns):
        missing_cols = required_columns - set(results_df.columns)
        raise ValueError(f"Missing required columns in the DataFrame: {missing_cols}")

    # Set up the figure size and color palette
    plt.figure(figsize=(15, 8))
    colors = Set1_9.mpl_colors

    # Plot Total Cost comparison
    plt.subplot(1, 3, 1)  # Create subplot 1 out of 3
    results_df_sort = results_df.sort_values(by='Total Cost(1000)')  # Sort DataFrame by Total Cost
    plt.bar(results_df_sort['Model'], results_df_sort['Total Cost(1000)'], color=colors[0])  # Plot bar chart
    plt.xlabel('Model')  # Set x-axis label
    plt.ylabel('Total Cost per 1000 docs')  # Set y-axis label
    plt.title('Total Cost Comparison (1000 docs)\n (Lowest is best)')  # Set plot title
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Plot Time Length comparison
    plt.subplot(1, 3, 2)  # Create subplot 2 out of 3
    results_df_sort = results_df.sort_values(by='Time Length')  # Sort DataFrame by Time Length
    plt.bar(results_df_sort['Model'], results_df_sort['Time Length'], color=colors[1])  # Plot bar chart
    plt.xlabel('Model')  # Set x-axis label
    plt.ylabel('Time Length (s)')  # Set y-axis label
    plt.title('Time Length Comparison\n (Lowest is best)')  # Set plot title
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Plot Summary Score comparison
    plt.subplot(1, 3, 3) # Create subplot 3 out of 3
    results_df_sort = results_df.sort_values(by='Summary Score', ascending=False)  # Sort DataFrame by Summary Score
    plt.bar(results_df_sort['Model'], results_df_sort['Summary Score'], color=colors[2])  # Plot bar chart
    plt.xlabel('Model')  # Set x-axis label
    plt.ylabel('Summary Score')  # Set y-axis label
    plt.title('Summary Score Comparison\n (Highest is best)')  # Set plot title
    plt.ylim(bottom=0, top=5)  # Set y-axis limits
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Adjust layout for better presentation
    plt.tight_layout()
    # Save the plot as an image in the reports directory
    plt.savefig("reports/graph.png")
    # Close the plot to free up memory
    plt.close()


def plot_model_performance_comparisons(results_df):
    """
    Plots comparisons between different models based on specified metrics in a grouped bar chart,
    with each metric performance displayed as a separate series.
    
    :param results_df: A pandas DataFrame containing the results of model comparisons.
    :return: None
    """
    
    # Check if input is a pandas DataFrame
    if not isinstance(results_df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame.")
        
    # Define required columns for comparison (adjust according to actual data)
    required_columns = {'model_name', 'model_completeness_score', 'model_flow_score',
                        'model_structure_score', 'model_conciseness_score',
                        'model_clarity_score', 'model_objectivity_score',
                        'model_tone_score', 'model_task_score'}
    
    # Check if required columns are present in the DataFrame
    if not required_columns.issubset(results_df.columns):
        missing_cols = required_columns - set(results_df.columns)
        raise ValueError(f"Missing required columns in the DataFrame: {missing_cols}")
    
    # Prepare data for plotting
    metrics = list(required_columns - {'model_name'})  # Exclude model_name from metrics
    
    models = results_df['model_name'].tolist()

    # Create color map based on number of metrics 
    colors_list = plt.get_cmap('Set1', 9).colors
    
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(12 + n_metrics, 8))  # Adjust figure size dynamically based on number of metrics
    bar_width = 0.05  # Adjust bar width for clarity
    
    for i, model in enumerate(models):
        positions = np.arange(len(metrics)) + i * (bar_width + 0.02)  # Positioning each group of bars
        
        scores = results_df[results_df['model_name'] == model][metrics].values.flatten().astype(int)
        rects = ax.bar(positions, scores, bar_width, label=model, color=colors_list[i % len(colors_list)])
        ax.bar_label(rects)
        
     # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Scores \n (Highest is best)')
     
    ax.set_xticks(np.arange(len(metrics)) + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
     
    plt.xticks(rotation=90)  # Rotate metric names for better visibility
    
    # Including model names within x-tick labels for clarity 
   # This requires custom formatting to intersperse metric names with model names dynamically.
    #ax.set_xticklabels([f'{metric}\n' + '\n'.join(models) for metric in metrics], ha='center')
    
    # Remove model names from x-tick labels
    ax.set_xticks(np.arange(len(metrics)) + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Models")
     
    plt.subplots_adjust(right=0.75)  # Adjust right edge to accommodate legend
    plt.ylim(bottom=0)  # Set y-axis bottom limit
    plt.tight_layout()
    plt.savefig("reports/rubric_graph.png")
     
    plt.close()

def plot_rag_comparisons(results_df):
    """
    Plots comparisons between different models based on specified metrics.

    :param results_df: A pandas DataFrame containing the results of model comparisons.
    :return: None
    """
    # Check if input is a pandas DataFrame
    if not isinstance(results_df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame.")

    # Define required columns for comparison
    required_columns = {'Model', 'Total Embedding Cost(1000)', 'Total LLM Cost(1000)', 'Time Length', 'Score'}
    # Check if required columns are present in the DataFrame
    if not required_columns.issubset(results_df.columns):
        missing_cols = required_columns - set(results_df.columns)
        raise ValueError(f"Missing required columns in the DataFrame: {missing_cols}")

    # Set up the figure size and color palette
    plt.figure(figsize=(15, 8))
    colors = Set1_9.mpl_colors

    # Plot Total Cost comparison
    plt.subplot(1, 4, 1)  # Create subplot 1 out of 4
    results_df_sort = results_df.sort_values(by='Total Embedding Cost(1000)')  # Sort DataFrame by Total Cost
    plt.bar(results_df_sort['Model'], results_df_sort['Total Embedding Cost(1000)'], color=colors[0])  # Plot bar chart
    plt.xlabel('Model')  # Set x-axis label
    plt.ylabel('Total Cost per 1000 embeddings')  # Set y-axis label
    plt.title('Total Cost Comparison (1000 embeddings)\n (Lowest is best)')  # Set plot title
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Plot Total Cost comparison
    plt.subplot(1, 4, 2)  # Create subplot 2 out of 4
    results_df_sort = results_df.sort_values(by='Total LLM Cost(1000)')  # Sort DataFrame by Total Cost
    plt.bar(results_df_sort['Model'], results_df_sort['Total LLM Cost(1000)'], color=colors[0])  # Plot bar chart
    plt.xlabel('Model')  # Set x-axis label
    plt.ylabel('Total Cost per 1000 llm invocations')  # Set y-axis label
    plt.title('Total Cost Comparison (1000 invocations)\n (Lowest is best)')  # Set plot title
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Plot Time Length comparison
    plt.subplot(1, 4, 3)  # Create subplot 3 out of 4
    results_df_sort = results_df.sort_values(by='Time Length')  # Sort DataFrame by Time Length
    plt.bar(results_df_sort['Model'], results_df_sort['Time Length'], color=colors[1])  # Plot bar chart
    plt.xlabel('Model')  # Set x-axis label
    plt.ylabel('Time Length (s)')  # Set y-axis label
    plt.title('Time Length Comparison\n (Lowest is best)')  # Set plot title
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Plot Summary Score comparison
    plt.subplot(1, 4, 4) # Create subplot 4 out of 4
    results_df_sort = results_df.sort_values(by='Score', ascending=False)  # Sort DataFrame by Summary Score
    plt.bar(results_df_sort['Model'], results_df_sort['Score'], color=colors[2])  # Plot bar chart
    plt.xlabel('Model')  # Set x-axis label
    plt.ylabel('Score')  # Set y-axis label
    plt.title('Score Comparison\n (Highest is best)')  # Set plot title
    plt.ylim(bottom=0, top=1)  # Set y-axis limits
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Adjust layout for better presentation
    plt.tight_layout()
    # Save the plot as an image in the reports directory
    plt.savefig("reports/graph.png")
    # Close the plot to free up memory
    plt.close()


def plot_rag_performance_comparisons(results_df):
    """
    Plots comparisons between different models based on specified metrics in a grouped bar chart,
    with each metric performance displayed as a separate series.
    
    :param results_df: A pandas DataFrame containing the results of model comparisons.
    :return: None
    """
    
    # Check if input is a pandas DataFrame
    if not isinstance(results_df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame.")
        
    # Define required columns for comparison (adjust according to actual data)
    required_columns = {'model_name', 'faithfulness', 'answer_relevancy',
                        'context_precision', 'context_recall',
                        'context_entity_recall', 'answer_similarity',
                        'answer_correctness', 'harmfulness', 'maliciousness', 
                        'coherence', 'correctness', 'conciseness'}
    
    # Check if required columns are present in the DataFrame
    if not required_columns.issubset(results_df.columns):
        print(results_df)
        missing_cols = required_columns - set(results_df.columns)
        raise ValueError(f"Missing required columns in the DataFrame: {missing_cols}")
    
    # Prepare data for plotting
    metrics = list(required_columns - {'model_name'})  # Exclude model_name from metrics
    
    models = results_df['model_name'].tolist()

    # Create color map based on number of metrics 
    colors_list = plt.get_cmap('Set1', 9).colors
    
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(12 + n_metrics, 8))  # Adjust figure size dynamically based on number of metrics
    bar_width = 0.05  # Adjust bar width for clarity
    
    for i, model in enumerate(models):
        positions = np.arange(len(metrics)) + i * (bar_width + 0.02)  # Positioning each group of bars
        
        scores = results_df[results_df['model_name'] == model][metrics].values.flatten().astype(float)
        rects = ax.bar(positions, scores, bar_width, label=model, color=colors_list[i % len(colors_list)])
        ax.bar_label(rects)
        
     # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Scores \n (Highest is best)')
     
    ax.set_xticks(np.arange(len(metrics)) + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
     
    plt.xticks(rotation=90)  # Rotate metric names for better visibility
    
    # Including model names within x-tick labels for clarity 
   # This requires custom formatting to intersperse metric names with model names dynamically.
    #ax.set_xticklabels([f'{metric}\n' + '\n'.join(models) for metric in metrics], ha='center')
    
    # Remove model names from x-tick labels
    ax.set_xticks(np.arange(len(metrics)) + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Models")
     
    plt.subplots_adjust(right=0.75)  # Adjust right edge to accommodate legend
    plt.ylim(bottom=0)  # Set y-axis bottom limit
    plt.tight_layout()
    plt.savefig("reports/rubric_graph.png")
     
    plt.close()