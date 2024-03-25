import joblib
import os

# Define the folder containing the joblib files
dir_path = input("Please enter the path to the folder containing the joblib files: ")

all_data = []

for file in os.listdir(dir_path):
    if file.endswith(".joblib"):
        data = joblib.load(os.path.join(dir_path, file))
        all_data.append(data)

all_data.sort(key=lambda x: x["rew"], reverse=True)
# Save the sorted list of dictionaries to a new joblib file
output_file = input("Please enter the name of the new joblib file to save the sorted data: ")
joblib.dump(all_data, output_file)
