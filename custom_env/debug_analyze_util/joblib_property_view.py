from joblib import load


def print_keys_and_lengths(joblib_file):
    data = load(joblib_file)
    # data is a list, print list size and first 5 elements
    print(f"Loaded {len(data)} elements.")
    # Print data dtype
    print(f"Data type: {type(data)}")
    for i, element in enumerate(data[:5]):
        print(element)


if __name__ == "__main__":
    joblib_file = input("Joblib file path: ")
    print_keys_and_lengths(joblib_file)