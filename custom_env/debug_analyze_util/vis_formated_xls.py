import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import datetime
from pathlib import Path
from tqdm import tqdm
import glob
import numpy as np


def extract_numeric_from_folder(folder_name):
    """
    Extract numeric part from folder name for sorting
    Expected format: tmp_[numeric_part]_suffix (e.g., tmp_202505201632592393178875_tt)

    Args:
        folder_name: String containing folder name

    Returns:
        int: Numeric part for sorting, or 0 if not found
    """
    # Pattern to extract numeric part between tmp_ and _suffix
    numeric_pattern = r'tmp_(\d+)_'
    match = re.search(numeric_pattern, folder_name)

    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 0
    return 0


def scan_and_read_xlsx(directory, subdir_prefix, file_prefix):
    """
    Scan directory for subdirectories with specific prefix and read xlsx files with specific file prefix

    Args:
        directory: Base directory path to scan
        subdir_prefix: Prefix to match subdirectories
        file_prefix: Prefix to match xlsx files (e.g., "output_Haoqiang_format")

    Returns:
        List of DataFrames with folder information
    """
    data_list = []

    # Find all subdirectories matching the prefix
    subdirs = [d for d in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, d)) and d.startswith(subdir_prefix)]

    print(f"Found {len(subdirs)} subdirectories matching prefix '{subdir_prefix}'")
    print(f"Subdirectories: {subdirs}")

    # Process each subdirectory with progress bar
    for subdir in tqdm(subdirs, desc="Reading xlsx files"):
        subdir_path = os.path.join(directory, subdir)

        # Find xlsx files with specific prefix in subdirectory
        all_xlsx_files = glob.glob(os.path.join(subdir_path, "*.xlsx"))
        matching_xlsx_files = [f for f in all_xlsx_files
                               if os.path.basename(f).startswith(file_prefix)]

        # Check if exactly one file matches the requirement
        if len(matching_xlsx_files) == 0:
            print(f"Error: No xlsx files with prefix '{file_prefix}' found in {subdir}")
            print(f"Available xlsx files: {[os.path.basename(f) for f in all_xlsx_files]}")
            exit(1)
        elif len(matching_xlsx_files) > 1:
            print(f"Error: Multiple xlsx files with prefix '{file_prefix}' found in {subdir}")
            print(f"Matching files: {[os.path.basename(f) for f in matching_xlsx_files]}")
            print("Expected exactly one file per subdirectory")
            exit(1)

        # Process the single matching file
        xlsx_file = matching_xlsx_files[0]
        try:
            # Read xlsx file
            df = pd.read_excel(xlsx_file)

            # Add folder name column for tracking
            df['Folder_Name'] = subdir

            # Filter data where Tag column equals 'tt'
            if 'Tag' in df.columns:
                df_filtered = df[df['Tag'] == 'tt'].copy()
                if not df_filtered.empty:
                    data_list.append(df_filtered)
                    print(f"Successfully processed: {os.path.basename(xlsx_file)} from {subdir}")
                else:
                    print(f"Warning: No records with Tag='tt' found in {xlsx_file}")
            else:
                print(f"Warning: No 'Tag' column found in {xlsx_file}")

        except Exception as e:
            print(f"Error reading {xlsx_file}: {str(e)}")
            exit(1)

    return data_list


def filter_and_sort_data(data_list):
    """
    Combine all dataframes and sort by numeric part extracted from folder names

    Args:
        data_list: List of DataFrames

    Returns:
        Combined and sorted DataFrame
    """
    if not data_list:
        return pd.DataFrame()

    # Combine all dataframes using concat for better performance
    combined_df = pd.concat(data_list, ignore_index=True)

    # Extract numeric part for sorting
    numeric_values = []
    for folder_name in combined_df['Folder Name']:
        numeric_value = extract_numeric_from_folder(folder_name)
        numeric_values.append(numeric_value)

    # Add numeric sort key column
    combined_df = combined_df.copy()  # Avoid fragmentation warning
    combined_df['Sort_Key'] = numeric_values

    # Remove rows with invalid numeric values (0) and sort
    valid_data = combined_df[combined_df['Sort_Key'] > 0].copy()
    sorted_data = valid_data.sort_values('Sort_Key').reset_index(drop=True)

    # Remove the temporary sort key column
    sorted_data = sorted_data.drop('Sort_Key', axis=1)

    print(f"Combined {len(data_list)} files into {len(sorted_data)} valid records")
    if len(sorted_data) > 0:
        first_folder = sorted_data['Folder_Name'].iloc[0]
        last_folder = sorted_data['Folder_Name'].iloc[-1]
        print(f"Folder range: {first_folder} to {last_folder}")

    return sorted_data


def plot_scatter_by_index(data, index_range, output_dir, prefix):
    """
    Plot scatter plots for columns within specified index range

    Args:
        data: DataFrame containing the data
        index_range: Tuple (start, end) for column indices
        output_dir: Directory to save plots
        prefix: Prefix for output files

    Returns:
        List of column names that were plotted
    """
    start_idx, end_idx = index_range
    plotted_columns = []

    # Create output directory if it doesn't exist
    img_dir = os.path.join(output_dir, f"{prefix}_img")
    os.makedirs(img_dir, exist_ok=True)

    # Get columns within the specified range
    columns = data.columns[start_idx:end_idx + 1]
    numeric_columns = []

    # Filter for numeric columns only
    for col in columns:
        if col not in ['Folder_Name', 'Sort_Key', 'Tag'] and pd.api.types.is_numeric_dtype(data[col]):
            numeric_columns.append(col)

    print(f"Plotting {len(numeric_columns)} numeric columns...")

    # Create a simple index for x-axis (representing chronological order)
    x_values = range(len(data))

    # Plot each numeric column with progress bar
    for col in tqdm(numeric_columns, desc="Creating scatter plots"):
        try:
            plt.figure(figsize=(10, 6))

            # Create scatter plot with index on x-axis (chronological order)
            plt.scatter(x_values, data[col], alpha=0.1, s=2, edgecolors='none')

            plt.title(f'{col} vs Chronological Order', fontsize=14, fontweight='bold')
            plt.xlabel('Chronological Order (Index)', fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot - handle special characters in column names
            sanitized_col_name = col.replace('/', '_').replace('\\', '_')
            plot_filename = f"{sanitized_col_name}.png"
            plot_path = os.path.join(img_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            plotted_columns.append(col)

        except Exception as e:
            print(f"Error plotting {col}: {str(e)}")

    return plotted_columns


def calculate_fom(data, delta_vout_indices, iq_index, C, ILOAD):
    """
    Calculate Figure of Merit (FoM) using the formula: (C * delta_VOUT * IQ) / (ILOAD)^2
    where delta_VOUT is the maximum of two specified columns

    Args:
        data: DataFrame containing the data
        delta_vout_indices: Tuple (col1_index, col2_index) for delta_VOUT calculation
        iq_index: Column index for IQ data
        C: Constant value C
        ILOAD: Constant value ILOAD

    Returns:
        DataFrame with added FoM column
    """
    try:
        col1_idx, col2_idx = delta_vout_indices
        columns = data.columns

        # Get column names
        col1_name = columns[col1_idx]
        col2_name = columns[col2_idx]
        iq_col_name = columns[iq_index]

        print(f"Calculating FoM using:")
        print(f"  delta_VOUT columns: {col1_name}, {col2_name}")
        print(f"  IQ column: {iq_col_name}")
        print(f"  C = {C}, ILOAD = {ILOAD}")

        # Calculate delta_VOUT as maximum of two columns
        delta_vout = np.maximum(data[col1_name], data[col2_name])

        # Calculate FoM: (C * delta_VOUT * IQ) / (ILOAD)^2
        fom = (C * delta_vout * data[iq_col_name]) / (ILOAD ** 2)

        # Add FoM column to dataframe
        data_with_fom = data.copy()
        data_with_fom['FoM'] = fom

        print(f"FoM calculated. Range: {fom.min():.2e} to {fom.max():.2e}")

        return data_with_fom

    except Exception as e:
        print(f"Error calculating FoM: {str(e)}")
        return data


def plot_fom_scatter(data, output_dir, prefix):
    """
    Plot scatter plot for FoM values

    Args:
        data: DataFrame containing FoM data
        output_dir: Directory to save plots
        prefix: Prefix for output files
    """
    try:
        if 'FoM' not in data.columns:
            print("No FoM column found for plotting")
            return

        img_dir = os.path.join(output_dir, f"{prefix}_img")

        # Create a simple index for x-axis (representing chronological order)
        x_values = range(len(data))

        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, data['FoM'], alpha=0.1, s=2, color='red', edgecolors='none')

        plt.title('Figure of Merit (FoM) vs Chronological Order', fontsize=14, fontweight='bold')
        plt.xlabel('Chronological Order (Index)', fontsize=12)
        plt.ylabel('FoM', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save FoM plot
        plot_path = os.path.join(img_dir, 'FoM.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print("FoM scatter plot saved successfully")

    except Exception as e:
        print(f"Error plotting FoM: {str(e)}")


def save_combined_data(data, output_path, prefix):
    """
    Save combined and processed data to Excel file

    Args:
        data: DataFrame to save
        output_path: Directory to save the file
        prefix: Prefix for filename
    """
    try:
        output_file = os.path.join(output_path, f"{prefix}_combined_data.xlsx")

        # Save to Excel file
        data.to_excel(output_file, index=False)

        print(f"Combined data saved to: {output_file}")
        print(f"Data shape: {data.shape}")

    except Exception as e:
        print(f"Error saving combined data: {str(e)}")


def main():
    """
    Main processing function
    """
    print("Starting batch visualization process...")

    # Step 1: Scan and read xlsx files
    print("\nStep 1: Scanning and reading xlsx files...")
    data_list = scan_and_read_xlsx(directory_path, subdir_prefix, xlsx_file_prefix)

    if not data_list:
        print("No valid data found. Exiting.")
        return

    # Step 2: Filter and sort data
    print("\nStep 2: Filtering and sorting data...")
    sorted_data = filter_and_sort_data(data_list)

    if sorted_data.empty:
        print("No valid data after filtering. Exiting.")
        return

    # Step 3: Plot scatter plots for specified index range
    print(f"\nStep 3: Creating scatter plots for columns {index_start} to {index_end}...")
    plotted_columns = plot_scatter_by_index(sorted_data, (index_start, index_end),
                                            directory_path, subdir_prefix)

    # Step 4: Calculate FoM
    print("\nStep 4: Calculating Figure of Merit (FoM)...")
    data_with_fom = calculate_fom(sorted_data, (delta_vout_col1_index, delta_vout_col2_index),
                                  iq_column_index, C_value, ILOAD_value)

    # Step 5: Plot FoM scatter
    print("\nStep 5: Creating FoM scatter plot...")
    plot_fom_scatter(data_with_fom, directory_path, subdir_prefix)

    # Step 6: Save combined data
    print("\nStep 6: Saving combined data...")
    save_combined_data(data_with_fom, directory_path, subdir_prefix)

    print(f"\nBatch visualization completed successfully!")
    print(f"Results saved in: {directory_path}")
    print(f"Images saved in: {os.path.join(directory_path, f'{subdir_prefix}_img')}")

if __name__ == '__main__':
    # Fixed input parameters - modify these values as needed
    # directory_path = "/Users/hanwu/Downloads/run_select/Haoqiang"  # Directory path to scan
    # subdir_prefix = "Haoqiang_Batch32_Group8"  # Subdirectory prefix to match
    directory_path = "/data/share/train_data/Haoqiang/run_select"  # Directory path to scan
    subdir_prefix = "Haoqiang_Batch32_Block_7b412"  # Subdirectory prefix to match
    xlsx_file_prefix = "output_Haoqiang_format"  # XLSX file prefix to match
    index_start = 2  # Index range start (inclusive)
    index_end = 25  # Index range end (inclusive)
    delta_vout_col1_index = 19  # First column index for delta_VOUT calculation
    delta_vout_col2_index = 20  # Second column index for delta_VOUT calculation
    iq_column_index = 3  # IQ column index
    C_value = 2e-12  # Constant C value for FoM calculation
    ILOAD_value = 0.05  # Constant ILOAD value for FoM calculation

    # Run main processing
    main()