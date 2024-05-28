import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_data(file_path, column_name):
    """
    Plot a scatter plot of the specified column against the timestamp extracted from the 'Folder Name'.

    Args:
    file_path (str): The path to the processed Excel file.
    column_name (str): The name of the column to plot.
    """
    # Load the Excel file
    data = pd.read_excel(file_path)

    # Extract the timestamp from 'Folder Name' and convert it to datetime
    data['Timestamp'] = pd.to_datetime(data['Folder Name'].str.extract(r'(\d{14})')[0], format='%Y%m%d%H%M%S')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Timestamp'], data[column_name], alpha=0.6, s=1, color='blue', edgecolors='none')
    plt.title(f'Scatter Plot of {column_name} over Time')
    plt.xlabel('Timestamp')
    plt.ylabel(column_name)

    # Set x-ticks to only show the start and end time
    plt.xticks([data['Timestamp'].min(), data['Timestamp'].max()],
               [data['Timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                data['Timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')])

    plt.grid(True)
    # plt.show()

    # Save the plot as an image file
    plot_file_name = f'{column_name}_scatter_plot.png'
    # Save to same folder as the Excel file
    plt.savefig(os.path.join(os.path.dirname(file_path), plot_file_name))
    plt.savefig(plot_file_name)


# Example usage
file_path = '/Users/hanwu/Downloads/Log_N65/UM_output/final_combined_output.xlsx'
column_name_list = ['Reward', 'DC_IQ', 'Load_Reg_loadReg', 'Line_Reg_100m_lineReg', 'Line_Reg_1m_lineReg',
                    'Trans_1_2V_overShoot', 'Trans_1_2V_underShoot', 'Trans_0_75V_overShoot', 'Trans_0_75V_underShoot',
                    'Trans_Line_Reg_overShoot', 'Trans_Line_Reg_underShoot', 'PSR_psr_100', 'PSR_psr_1k', 'PSR_psr_10k',
                    'PSR_psr_100k', 'PSR_psr_1M']
for column_name in column_name_list:
    plot_data(file_path, column_name)
