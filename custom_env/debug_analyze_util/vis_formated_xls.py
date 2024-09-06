import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def plot_data(data, column_name):
    """
    Plot a scatter plot of the specified column against the sorted index.

    Args:
    data (pd.DataFrame): The loaded Excel data.
    column_name (str): The name of the column to plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data.index, data[column_name], alpha=0.6, s=1, color='blue', edgecolors='none')
    ax.set_title(f'Scatter Plot of {column_name} over Time')
    ax.set_xlabel('Time Index')
    ax.set_ylabel(column_name)
    ax.set_xticks([0, len(data) - 1])
    ax.set_xticklabels([data['Timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
                        data['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')])
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


def filter_by_tag(data):
    """
    Filter the data based on the Tag column.

    Args:
    data (pd.DataFrame): The loaded Excel data.

    Returns:
    pd.DataFrame: Filtered data.
    """
    if 'Tag' not in data.columns:
        use_all = input("Tag column not found. Do you want to proceed with all data? (yes/no): ").lower()
        return data if use_all == 'yes' or use_all == 'y' else None

    default_tags = ['ff', 'ss', 'fs', 'tt']
    use_default = input(f"Do you want to use default tags {default_tags}? (yes/no): ").lower()

    if use_default == 'yes' or use_default == 'y':
        tags = default_tags
    else:
        tags = input("Enter tags to filter (comma-separated): ").split(',')
        tags = [tag.strip() for tag in tags]

    return data[data['Tag'].isin(tags)]


def sort_data_by_path(data):
    """
    Sort the data based on the full file path.

    Args:
    data (pd.DataFrame): The loaded Excel data.

    Returns:
    pd.DataFrame: Sorted data.
    """
    print("Sorting data based on file paths...")

    # Combine 'Folder Name' and 'File Name' to create a full path
    data['Full Path'] = data['Folder Name'].astype(str) + '/' + data['File Name'].astype(str)

    # Sort the data based on the full path with progress bar
    sorted_data = data.sort_values('Full Path')

    # Reset the index after sorting
    sorted_data = sorted_data.reset_index(drop=True)

    return sorted_data


# Example usage
file_path = input('Enter the path to the processed Excel file: ')

# Load the Excel file with progress bar
data = pd.read_excel(file_path)

# Filter data by Tag
filtered_data = filter_by_tag(data)
if filtered_data is None:
    print("Data filtering cancelled. Exiting.")
    exit()

# Extract the timestamp from 'Folder Name' and convert it to datetime
print(f"Extracting timestamps from 'Folder Name'...")
filtered_data['Timestamp'] = pd.to_datetime(filtered_data['Folder Name'].str.extract(r'(\d{14})')[0],
                                            format='%Y%m%d%H%M%S')

# Sort the data based on the full file path
sorted_data = sort_data_by_path(filtered_data)

# Get all column names except 'Folder Name', 'Timestamp', 'Tag', 'File Name', and 'Full Path'
all_columns = [col for col in sorted_data.columns if
               col not in ['Folder Name', 'Timestamp', 'Tag', 'File Name', 'Full Path']]

# Let the user select columns
column_name_list = select_columns(all_columns)

# Create a directory to store the plots
plot_dir = os.path.join(os.path.dirname(file_path), 'scatter_plots_' + os.path.basename(file_path))
os.makedirs(plot_dir, exist_ok=True)

# Plot the data and save each plot as a separate image file
for column_name in tqdm(column_name_list, desc="Generating plots"):
    fig = plot_data(sorted_data, column_name)
    plot_file_name = f'{column_name}_scatter_plot.png'
    plot_file_path = os.path.join(plot_dir, plot_file_name)
    fig.savefig(plot_file_path)
    plt.close(fig)

print("All plots have been generated and saved in the 'scatter_plots' directory.")