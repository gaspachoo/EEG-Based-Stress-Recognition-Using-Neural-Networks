import numpy as np
import pandas as pd
from scipy.io import loadmat
import os

def load_dataset(data_dir, labels_filename):
    subjects = range(1, 41)
    trials = range(1, 4)

    # Define mapping from Excel task names to folder names
    task_mapping = {'Maths': 'Arithmetic', 'Symmetry': 'Mirror_image', 'Stroop': 'Stroop'}
    tasks_excel = list(task_mapping.keys())   # ['Maths', 'Symmetry', 'Stroop']

    labels_filepath = f'./Data/{labels_filename}'
    df = pd.read_excel(labels_filepath, header=[0, 1])
    
    # Initialize lists with the correct structure (40 subjects, 9 entries per subject)
    labels = []
    data = []

    for subject in subjects:
        row = df[df[('Subject No.', 'Unnamed: 0_level_1')] == subject].squeeze()
        
        subject_labels = []
        subject_data = []

        for trial in trials:
            trial_name = f'Trial_{trial}'

            for excel_task in tasks_excel:  # Using Excel task names
                # Retrieve the score
                score = int(row[(trial_name, excel_task)])
                subject_labels.append(score)

                # Get the corresponding folder name
                task_folder = task_mapping[excel_task]

                # Load the EEG file
                filename = f"{task_folder}_sub_{subject}_trial{trial}.mat"
                filepath = f'./Data/{data_dir}/{filename}'
                
                # Before checking os.path.exists(filepath)
                if os.path.exists(filepath):
                    mat = loadmat(filepath)
                    eeg_data = mat['Data'] if data_dir == 'raw_data' else mat['Clean_data']
                    subject_data.append(eeg_data)
                else:
                    print(f"File {filename} not found. Skipping.",flush=True)
                    subject_data.append(None)  # Placeholder to maintain structure

        labels.append(subject_labels)
        data.append(subject_data)

    labels = np.array(labels)  # Shape: (40, 9)
    data = np.array(data, dtype=object)  # Shape: (40, 9, channels, time)

    return data, labels

data_folder = 'filtered_data'
labels_file = 'scales.xls'

load_dataset(data_folder,labels_file)