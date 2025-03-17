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


import numpy as np
import pandas as pd
from scipy.io import loadmat
import os

def segment_eeg(eeg_data, segment_length=256):
    """
    Découpe les données EEG en segments de taille fixe.

    Args:
        eeg_data (numpy.ndarray): Données EEG de forme (channels, time)
        segment_length (int): Longueur du segment en échantillons

    Returns:
        numpy.ndarray: Données segmentées de forme (n_segments, channels, segment_length)
        int: Nombre de segments complets générés
    """
    num_samples = eeg_data.shape[1]
    num_segments = num_samples // segment_length  # Nombre total de segments complets
    segmented_data = eeg_data[:, :num_segments * segment_length]  # Tronque les échantillons restants
    segmented_data = segmented_data.reshape((eeg_data.shape[0], num_segments, segment_length))
    segmented_data = np.transpose(segmented_data, (1, 0, 2))  # Reshape vers (n_segments, channels, segment_length)

    return segmented_data, num_segments

def load_dataset2(data_dir, labels_filename, f_sampling=128, segment_duration=2):
    subjects = range(1, 41)
    trials = range(1, 4)

    # Mapping des tâches
    task_mapping = {'Maths': 'Arithmetic', 'Symmetry': 'Mirror_image', 'Stroop': 'Stroop'}
    tasks_excel = list(task_mapping.keys())

    labels_filepath = f'./Data/{labels_filename}'
    df = pd.read_excel(labels_filepath, header=[0, 1])

    segment_length = segment_duration * f_sampling  # 2 sec * 128 Hz = 256 échantillons

    # Initialisation des structures
    data = np.empty((40, 9), dtype=object)
    labels = np.empty((40, 9), dtype=object)

    for subject in subjects:
        row = df[df[('Subject No.', 'Unnamed: 0_level_1')] == subject].squeeze()
        
        for trial_idx, trial in enumerate(trials):
            trial_name = f'Trial_{trial}'

            for task_idx, excel_task in enumerate(tasks_excel):
                score = int(row[(trial_name, excel_task)])
                task_folder = task_mapping[excel_task]
                filename = f"{task_folder}_sub_{subject}_trial{trial}.mat"
                filepath = f'./Data/{data_dir}/{filename}'

                if os.path.exists(filepath):
                    mat = loadmat(filepath)
                    eeg_data = mat['Data'] if data_dir == 'raw_data' else mat['Clean_data']
                    
                    # Découpe en segments de 2 secondes
                    segmented_eeg, num_segments = segment_eeg(eeg_data, segment_length)

                    # Stockage des données et des labels
                    data[subject-1, trial_idx * 3 + task_idx] = segmented_eeg  # (num_segments, channels, time)
                    labels[subject-1, trial_idx * 3 + task_idx] = np.full((num_segments,), score)  # (num_segments,)

                    #print(f"Subject {subject}, Trial {trial}, Task {excel_task}: {num_segments} segments", flush=True)
                
                else:
                    print(f"File {filename} not found. Skipping.", flush=True)
                    data[subject-1, trial_idx * 3 + task_idx] = None
                    labels[subject-1, trial_idx * 3 + task_idx] = None

    return data, labels  # Shape: (40, 9) but each entry is (num_segments, channels, time) for data

data_folder = 'filtered_data'
labels_file = 'scales.xls'

data, labels = load_dataset2(data_folder, labels_file)
