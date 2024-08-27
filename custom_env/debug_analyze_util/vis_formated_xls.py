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


def select_columns(columns):
    """
    Display column names and let the user select a range of columns.

    Args:
    columns (list): List of column names.

    Returns:
    list: Selected column names.
    """
    while True:
        print("\nAvailable columns:")
        for i, col in enumerate(columns):
            print(f"{i}: {col}")

        start_index = int(input("Enter the starting column index: "))
        end_index = int(input("Enter the ending column index: "))

        selected_columns = columns[start_index:end_index + 1]

        print("\nYou have selected the following columns:")
        for i, col in enumerate(selected_columns, start=1):
            print(f"{i}. {col}")

        confirm = input("\nDo you confirm this selection? (yes/no): ").lower()
        if confirm == 'yes' or confirm == 'y':
            return selected_columns
        else:
            print("Selection cancelled. Please try again.")


# Example usage
file_path = input('Enter the path to the processed Excel file: ')

# Load the Excel file
print(f"Loading data from file: {file_path}")
data = pd.read_excel(file_path)

# Extract file name
file_name = os.path.basename(file_path)

# Extract the timestamp from 'Folder Name' and convert it to datetime
print(f"Extracting timestamps from 'Folder Name'...")
data['Timestamp'] = pd.to_datetime(data['Folder Name'].str.extract(r'(\d{14})')[0], format='%Y%m%d%H%M%S')

# Get all column names except 'Folder Name' and 'Timestamp'
all_columns = [col for col in data.columns if col not in ['Folder Name', 'Timestamp']]

# Let the user select columns
column_name_list = select_columns(all_columns)

# Create a directory to store the plots
plot_dir = os.path.join(os.path.dirname(file_path), 'scatter_plots_' + file_name)
os.makedirs(plot_dir, exist_ok=True)

# Plot the data and save each plot as a separate image file
for column_name in column_name_list:
    print(f"Drawing scatter plot for column: {column_name}...")
    fig = plot_data(data, column_name)
    plot_file_name = f'{column_name}_scatter_plot.png'
    plot_file_path = os.path.join(plot_dir, plot_file_name)
    fig.savefig(plot_file_path)
    plt.close(fig)

print("All plots have been generated and saved in the 'scatter_plots' directory.")