import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from exp_utils import OUT_DIR, BASE_DIR

MEM_PLOT_DIR = f"{BASE_DIR}/peak_mem_plots"

# Specify the column names for the CSV
columns = [
    'model_name', 'batch_size', 'seq_len', 'image_size', 'denoising_steps',
    'precision', 'ac_mode', 'sac_algo', 'peak_memory', 'estimation_time'
]

# Read the CSV file (replace 'data.csv' with your file path)
file_path = f"{OUT_DIR}/memory_estimation_H100.csv"  # Update with the correct file path
data = pd.read_csv(file_path, names=columns)

# Convert peak_memory from bytes to GiB
data['peak_memory_gib'] = data['peak_memory'] / (1024 ** 3)

# Get unique model names
model_names = data['model_name'].unique()

# AC modes and their labels
ac_modes = ['none', 'full', 'auto']
sac_algos = ['optimal', 'knapsack', 'greedy']
bar_labels = ['No AC', 'Full AC', 'Auto SAC[optimal]', 'Auto SAC[knapsack]', 'Auto SAC[greedy]']

# Colors and patterns for bars
bar_colors = ['#D3D3D3', '#FFC0CB', '#ADD8E6', '#C6F4C6', '#C7B8EA']
# bar_colors = ['black', 'maroon', 'lavender', 'purple', 'violet']
bar_patterns = ['', '.', '\\', 'x', '/']

# Loop over each model name
for model_name in model_names:
    model_data = data[data['model_name'] == model_name]
    
    # Unique configurations
    configs = model_data[['batch_size', 'seq_len', 'image_size', 'denoising_steps', 'precision']].drop_duplicates()
    
    # Sort configurations by peak_memory of ac_mode 'none'
    configs['peak_memory_none'] = configs.apply(
        lambda row: model_data[
            (model_data['batch_size'] == row['batch_size']) &
            (model_data['seq_len'] == row['seq_len']) &
            (model_data['image_size'] == row['image_size']) &
            (model_data['denoising_steps'] == row['denoising_steps']) &
            (model_data['precision'] == row['precision']) &
            (model_data['ac_mode'] == 'none')
        ]['peak_memory_gib'].values[0] if not model_data.empty else 0,
        axis=1
    )
    configs = configs.sort_values(by='peak_memory_none', ascending=False)
    
    # Plot data
    num_configs = len(configs)
    x_positions = np.arange(num_configs) * 0.75
    bar_width = 0.1
    bar_offsets = np.arange(len(bar_labels)) * bar_width

    plt.figure(figsize=(6, 8))

    for i, (_, config) in enumerate(configs.iterrows()):
        config_data = model_data[
            (model_data['batch_size'] == config['batch_size']) &
            (model_data['seq_len'] == config['seq_len']) &
            (model_data['image_size'] == config['image_size']) &
            (model_data['denoising_steps'] == config['denoising_steps']) &
            (model_data['precision'] == config['precision'])
        ]
        
        # Extract memory data for each AC mode and algorithm
        memory_values = []
        for mode in ac_modes:
            if mode == 'auto':
                for algo in sac_algos:
                    memory = config_data[
                        (config_data['ac_mode'] == mode) &
                        (config_data['sac_algo'] == algo)
                    ]['peak_memory_gib']
                    memory_values.append(memory.values[0] if not memory.empty else 0)
            else:
                memory = config_data[config_data['ac_mode'] == mode]['peak_memory_gib']
                memory_values.append(memory.values[0] if not memory.empty else 0)

        # Plot bars for this configuration
        for j, memory in enumerate(memory_values):
            plt.bar(x_positions[i] + bar_offsets[j], memory, width=bar_width, 
                    color=bar_colors[j], hatch=bar_patterns[j], label=bar_labels[j] if i == 0 else "")

    # Add horizontal lines for memory budget and GPU memory limit
    plt.axhline(y=68, color='orange', linestyle='--', label='Memory Budget (68 GiB)')
    plt.axhline(y=80, color='red', linestyle='--', label='GPU Memory Limit (80 GiB)')

    # Customize plot
    plt.xticks(x_positions + bar_width * (len(bar_labels) - 1) / 2, [f"C{i+1}" for i in range(num_configs)], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Peak Memory (GiB)", fontsize=14)
    plt.xlabel("Configurations", fontsize=14)
    # plt.title(f"Peak Memory Usage per Configuration for {model_name}")
    plt.legend(title="AC Modes", loc='upper right', title_fontsize=14, fontsize=12)
    plt.tight_layout()

    # Save plot as a PDF with tight layout and no margins
    output_file = f"{MEM_PLOT_DIR}/{model_name}_peak_memory.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight', pad_inches=0)

    # Close the figure to free memory
    plt.close()
