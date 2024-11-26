import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
from exp_utils import OUT_DIR, BASE_DIR
gpu_type="H100"
BREAKDOWN_DIR = f"{BASE_DIR}/memory_breakdown_plots"
# Define column titles
column_titles = ['model_name', 'batch_size', 'seq_len', 'image_size', 'denoising_steps', 'precision', 'ac', 
                 'Parameter', 'Buffer', 'Gradient', 'Activation', 'Temp', 'Optstate', 'Other', 'Total']
df = pd.read_csv(f'{OUT_DIR}/memory_estimation_snapshot_{gpu_type}.csv', names=column_titles)
# Define custom lighter colors
colors = ['#ADD8E6', '#C6F4C6', '#FFC0CB', '#C7B8EA', '#F5DEB3', '#FFD7BE', '#D3D3D3']
# Group by model_name
for model_name, group in df.groupby('model_name'):
    # Plot stacked bar chart
# Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(6, 6))
    group[['Parameter', 'Gradient', 'Activation', 'Optstate', 'Temp', 'Buffer', 'Other']].div(2**30).plot(
        kind='bar', stacked=True, ax=ax, color=colors, width=0.2)
    # Add horizontal dotted line at 40 GiB
    ax.axhline(40, color='black', linestyle=':', linewidth=1)

    # Set custom y-axis limits
    ax.set_ylim(0, 100)

    # Set labels and title
    ax.set_xlabel('Config')
    ax.set_ylabel('Memory (GiB)')
    ax.set_title(model_name)

    # Set xtick labels
    ax.set_xticklabels([f'Config {i}' for i in range(len(group))])

    # Add legend
    ax.legend(title='Components', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save plot as PDF
    plt.tight_layout()
    plt.savefig(f'{BREAKDOWN_DIR}/{model_name}_peak_memory_breakdown.pdf', bbox_inches='tight')

    # Close figure
    plt.close(fig)