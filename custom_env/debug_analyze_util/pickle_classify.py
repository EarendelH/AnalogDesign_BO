import os
import pickle
import shutil


def count_sim_params(pickle_file):
    """Count the number of simulation parameters in a pickle file."""
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    sim_results = data['initial_data']['sim_result']
    return len(sim_results)


def classify_pickle_files(source_dir):
    """Classify pickle files by the number of simulation parameters."""
    for filename in os.listdir(source_dir):
        if filename.endswith('.pkl'):
            pickle_path = os.path.join(source_dir, filename)
            num_params = count_sim_params(pickle_path) + 1
            target_dir = os.path.join(source_dir, f'specs_{num_params}')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.move(pickle_path, os.path.join(target_dir, filename))
            print(f'Moved {filename} to {target_dir}')


if __name__ == '__main__':
    source_dir = input('Path to directory with pickle files: ')
    classify_pickle_files(source_dir)
