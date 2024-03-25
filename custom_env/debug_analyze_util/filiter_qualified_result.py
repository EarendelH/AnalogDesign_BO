from joblib import load, dump
import sys


def filter_data(input_file, threshold):

    data = load(input_file)
    filtered_data = []

    for item in data:
        if item['rew'] >= threshold:
            print(f"Filtered data: {item}")
            filtered_data.append(item)

    save = input("Do you want to save the filtered data to a new joblib file? (y/n): ")
    if save.lower() == 'y':
        output_file = input("Please enter the name of the new joblib file: ")
        dump(filtered_data, output_file)
        print(f"Filtered data has been saved to {output_file}.")
    else:
        print("Filtered data has not been saved.")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_joblib_file>")
        sys.exit(1)
    threshold_value = input("Please enter the threshold value for filtering: ")
    input_file = sys.argv[1]
    filter_data(input_file, float(threshold_value))
