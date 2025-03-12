import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_dataset(data_dir):
    data = []
    subjects = range(1,41)
    trials = range(1,4)
    tasks = ['Arithmetic','Mirror_image','Stroop']

    for subject in subjects:
        for task in tasks:
            for trial in trials:
                filename = f"{task}_sub_{subject}_trial{trial}.mat"
                filepath = './Data/' + data_dir + '/' + filename
                
                try:
                    mat = loadmat(filepath)
                    if data_dir == 'raw_data' :
                        eeg_data = mat['Data']                       
                    else:
                        eeg_data = mat['Clean_data']  # Shape: (channels, time)
                    
                    data.append(eeg_data)
                except FileNotFoundError:
                    print(f"File {filename} not found. Skipping.")
    
    data = np.array(data)  # Shape: (samples, channels, time)
    
    return data

def load_labels(filename):
    tasks_mapping = {'Maths': 'Arithmetic', 'Symmetry': 'Mirror_image', 'Stroop': 'Stroop'}
    trials = [1, 2, 3]
    tasks_order = ['Maths', 'Symmetry', 'Stroop']
    subjects = range(1,41)
    
    # Load with multi-level headers
    filepath = './Data/' + filename
    df = pd.read_excel(filepath, header=[0, 1])
    
    # Initialize labels array: (subjects, trials, tasks)
    labels_array = np.zeros((40, 3, 3))

    for subj_idx, subject in enumerate(subjects):
        # Select the row for the current subject
        row = df.loc[df[('Subject No.', 'Unnamed: 0_level_1')] == subject]

        if row.empty:
            raise ValueError(f"Subject {subject} not found in the file.")

        row = row.squeeze()  # Convert single-row DataFrame to Series

        for trial_idx, trial_num in enumerate(trials):
            trial_name = f'Trial_{trial_num}'

            for task_idx, excel_task_name in enumerate(tasks_order):
                score = row[(trial_name, excel_task_name)]
                labels_array[subj_idx, trial_idx, task_idx] = int(score)

    return labels_array.reshape(-1)

def split_data(data, labels, test_size=0.2, random_state=None):
    
    if random_state is not None:
        np.random.seed(random_state)  # Ensure reproducibility

    indices = np.arange(len(data))
    np.random.shuffle(indices)

    split_idx = int(len(data) * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = data[train_indices]
    y_train = labels[train_indices]
    X_test = data[test_indices]
    y_test = labels[test_indices]

    return X_train, X_test, y_train, y_test