import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from exp_utils import OUT_DIR, BASE_DIR

OPT_PLOT_DIR = f"{BASE_DIR}/optimization_times_plots"

# Specify the column names for the CSV
columns = [
    'model_name', 'batch_size', 'seq_len', 'image_size', 'denoising_steps',
    'precision', 'ac_mode', 'sac_algo', 'ilp_time', 'policy_time'
]

# Read the CSV file (replace 'data.csv' with your file path)
file_path = f"{OUT_DIR}/auto_sac_H100.csv" 
data = pd.read_csv(file_path, names=columns)

data['policy_time'] = data['policy_time'].astype(float)

# Get unique model names
model_names = data['model_name'].unique()

# AC modes and their labels
sac_algos = ['optimal', 'knapsack', 'greedy']
bar_labels = ['Auto SAC[optimal]', 'Auto SAC[knapsack]', 'Auto SAC[greedy]']

# Colors and patterns for bars
bar_colors = ['#ADD8E6', '#C6F4C6', '#C7B8EA']
# bar_colors = ['lavender', 'purple', 'violet']
bar_patterns = ['\\', 'x', '/']

# Loop over each model name
for model_name in model_names:
    model_data = data[data['model_name'] == model_name]
    
    # Unique configurations
    configs = model_data[['batch_size', 'seq_len', 'image_size', 'denoising_steps', 'precision']].drop_duplicates()
        
    # Plot data
    num_configs = len(configs)
    x_positions = np.arange(num_configs) * 0.75
    bar_width = 0.1
    bar_offsets = np.arange(len(bar_labels)) * bar_width

    plt.figure(figsize=(6, 6))

    for i, (_, config) in enumerate(configs.iterrows()):
        config_data = model_data[
            (model_data['batch_size'] == config['batch_size']) &
            (model_data['seq_len'] == config['seq_len']) &
            (model_data['image_size'] == config['image_size']) &
            (model_data['denoising_steps'] == config['denoising_steps']) &
            (model_data['precision'] == config['precision'])
        ]
        
        # Extract runtimes data for each AC mode and algorithm
        opt_times = []

        for algo in sac_algos:
            policy_times = config_data[
                (config_data['sac_algo'] == algo)
            ]['policy_time']
            opt_times.append(policy_times.values[0] if not policy_times.empty else 0)
        # Plot bars for this configuration
        for j, opt_time in enumerate(opt_times):
            plt.bar(x_positions[i] + bar_offsets[j], opt_time, width=bar_width, 
                    color=bar_colors[j], hatch=bar_patterns[j], label=bar_labels[j] if i == 0 else "")


    # Customize plot
    plt.xticks(x_positions + bar_width * (len(bar_labels) - 1) / 2, [f"C{i+1}" for i in range(num_configs)], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Optimization Time (ms)", fontsize=14)
    plt.xlabel("Configurations", fontsize=14)
    # plt.title(f"Peak Memory Usage per Configuration for {model_name}")
    plt.legend(title="SAC Algorithms", loc='best', title_fontsize=14, fontsize=12)
    plt.tight_layout()

    # Save plot as a PDF with tight layout and no margins
    output_file = f"{OPT_PLOT_DIR}/{model_name}_opt_times.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight', pad_inches=0)

    # Close the figure to free runtimes
    plt.close()
