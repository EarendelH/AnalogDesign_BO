import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_data(data, column_name):
    """
    Plot a scatter plot of the specified column against the timestamp extracted from the 'Folder Name'.

    Args:
    data (pd.DataFrame): The loaded Excel data.
    column_name (str): The name of the column to plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data['Timestamp'], data[column_name], alpha=0.6, s=1, color='blue', edgecolors='none')
    ax.set_title(f'Scatter Plot of {column_name} over Time')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(column_name)
    ax.set_xticks([data['Timestamp'].min(), data['Timestamp'].max()])
    ax.set_xticklabels([data['Timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                        data['Timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')])
    ax.grid(True)
    plt.tight_layout()
    return fig


# Example usage
file_path = input('Enter the path to the processed Excel file: ')
column_name_list = ['Reward', 'DC_IQ', 'Load_Reg_loadReg', 'Line_Reg_100m_lineReg', 'Line_Reg_1m_lineReg',
                    'Trans_1_2V_overShoot', 'Trans_1_2V_underShoot', 'Trans_0_75V_overShoot', 'Trans_0_75V_underShoot',
                    'Trans_Line_Reg_overShoot', 'Trans_Line_Reg_underShoot', 'PSR_psr_100', 'PSR_psr_1k', 'PSR_psr_10k',
                    'PSR_psr_100k', 'PSR_psr_1M']

# Load the Excel file
print(f"Loading data from file: {file_path}")
data = pd.read_excel(file_path)

# Extract the timestamp from 'Folder Name' and convert it to datetime
print(f"Extracting timestamps from 'Folder Name'...")
data['Timestamp'] = pd.to_datetime(data['Folder Name'].str.extract(r'(\d{14})')[0], format='%Y%m%d%H%M%S')

# Create a directory to store the plots
plot_dir = os.path.join(os.path.dirname(file_path), 'scatter_plots')
os.makedirs(plot_dir, exist_ok=True)

# Plot the data and save each plot as a separate image file
for column_name in column_name_list:
    print(f"Drawing scatter plot for column: {column_name}...")
    fig = plot_data(data, column_name)
    plot_file_name = f'{column_name}_scatter_plot.png'
    plot_file_path = os.path.join(plot_dir, plot_file_name)
    fig.savefig(plot_file_path)
    plt.close(fig)
