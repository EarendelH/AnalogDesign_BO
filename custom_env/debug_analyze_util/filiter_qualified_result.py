from joblib import load, dump
import sys
import pandas as pd


def check_values(data):
    for key, value in data.items():
        if value == 0.0:
            return False
        if key == 'DC_IQ' and value == 1.0:
            return False
    return True


def filter_data(input_pkl, threshold):

    data = load(input_pkl)
    filtered_data = []

    valid_data = []
    for item in data:
        if check_values(item['result']):
            valid_data.append(item)

    print(f"Valida data number: {len(valid_data)} extracted from {len(data)} data.")

    for item in valid_data:
        if item['rew'] >= threshold:
            # print(f"Filtered data: {item}")
            filtered_data.append(item)

    sorted_list = sorted(filtered_data, key=lambda x: x['rew'], reverse=True)

    save_joblib = input("Do you want to save the filtered data to a new joblib file? (y/n): ")
    if save_joblib.lower() == 'y':
        output_file = input("Please enter the name of the new joblib file: ")
        dump(sorted_list, output_file)
        print(f"Filtered data has been saved to {output_file}.")
    else:
        print("Filtered data has not been saved.")
        sys.exit(0)

    # Save csv file
    save_csv = input("Do you want to save the filtered data to a new csv file? (y/n): ")
    if save_csv.lower() == 'y':
        df = pd.DataFrame(sorted_list)
        output_csv_path = input("Please enter the name of the new csv file to save the sorted data: ")
        df.to_csv(output_csv_path, index=False)
    else:
        print("Filtered data has not been saved.")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_joblib_file>")
        sys.exit(1)
    threshold_value = input("Please enter the threshold reward value for filtering: ")
    input_file = sys.argv[1]
    filter_data(input_file, float(threshold_value))
