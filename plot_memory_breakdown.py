import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
from exp_utils import OUT_DIR, BASE_DIR
gpu_type="H100"
BREAKDOWN_DIR = f"{BASE_DIR}/memory_breakdown_plots"
# Define column titles
column_titles = ['model_name', 'batch_size', 'seq_len', 'image_size', 'denoising_steps', 'precision', 'ac', 'sac_algo',
                 'Parameter', 'Buffer', 'Gradient', 'Activation', 'Temp', 'Optstate', 'Other', 'Total']
full_df = pd.read_csv(f'{OUT_DIR}/memory_estimation_snapshot_{gpu_type}.csv', names=column_titles)
# Filter rows where 'ac' equals 'none'
df = full_df.loc[full_df['ac'] == 'none']
# Define custom lighter colors
colors = ['#ADD8E6', '#C6F4C6', '#FFC0CB', '#C7B8EA', '#F5DEB3', '#FFD7BE', '#D3D3D3']
# Group by 'model_name' and sort each group by 'Total'
sorted_groups = {}
for model_name, group in df.groupby('model_name'):
    sorted_groups[model_name] = group.sort_values(by='Total', ascending=False)
# Group by model_name
for model_name, group in sorted_groups.items():
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 8))
    group[['Parameter', 'Gradient', 'Activation', 'Optstate', 'Temp', 'Buffer', 'Other']].div(2**30).plot(
        kind='bar', stacked=True, ax=ax, color=colors, width=0.2)
    # Add horizontal lines for memory budget and GPU memory limit
    ax.axhline(y=68, color='orange', linestyle='--', label='Memory Budget (68 GiB)')
    ax.axhline(y=80, color='red', linestyle='--', label='GPU Memory Limit (80 GiB)')

    # # Set custom y-axis limits
    ax.set_ylim(0, 160)

    # Set labels and title
    ax.set_xlabel('Configurations', fontsize=14)
    ax.set_ylabel('Memory (GiB)', fontsize=14)

    # Set xtick labels
    ax.set_xticks(range(len(group)))  # Set positions for x-axis ticks
    ax.set_xticklabels([f'C{i+1}' for i in range(len(group))])  # Set x-axis tick labels
    ax.tick_params(axis='x', labelsize=14)  # Adjust x-axis tick font size
    ax.tick_params(axis='y', labelsize=14)  # Adjust y-axis tick font size

    # Add legend
    ax.legend(title='Components', loc='best', title_fontsize=14, fontsize=12)


    # Save plot as PDF
    plt.tight_layout()
    plt.savefig(f'{BREAKDOWN_DIR}/{model_name}_peak_memory_breakdown.pdf', bbox_inches='tight', pad_inches=0)

    # Close figure
    plt.close(fig)