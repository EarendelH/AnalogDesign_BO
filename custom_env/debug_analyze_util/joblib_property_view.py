from joblib import load


def print_keys_and_lengths(joblib_file):
    data = load(joblib_file)

    for key, values_list in data.items():
        print(f"{key}: Length: {len(values_list)}")


if __name__ == "__main__":
    joblib_file = input("Joblib file path: ")
    print_keys_and_lengths(joblib_file)