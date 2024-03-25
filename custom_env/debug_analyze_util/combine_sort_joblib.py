import joblib
import os

# Define the folder containing the joblib files
folder_path = input("Please enter the path to the folder containing the joblib files: ")

# Initialize an empty list to store the dictionaries
dict_list = []

# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a joblib file
    if file_name.endswith('.joblib'):
        # Construct the full path of the file
        file_path = os.path.join(folder_path, file_name)
        # Load the dictionary from the joblib file
        data = joblib.load(file_path)
        # Append the loaded dictionary to the list
        dict_list.append(data)

# Sort the list of dictionaries by the 'rew' value in descending order
sorted_dict_list = sorted(dict_list, key=lambda x: x['rew'], reverse=True)

# Save the sorted list of dictionaries to a new joblib file
output_file = input("Please enter the name of the new joblib file to save the sorted data: ")
joblib.dump(sorted_dict_list, output_file)
