import os
from scipy.io import loadmat, savemat
import cleaning_algos

def filter_all_data(method_name, input_dir):
    
    method_used = getattr(cleaning_algos, method_name)
    output_dir = f'../Data/{method_name}_filtered_data'
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all .mat files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")

            # Load EEG data from the .mat file
            mat = loadmat(filepath)
            if 'Data' not in mat:
                print(f"❌ 'Data' not found in {filename}. Skipping.")
                continue
            eeg_data = mat['Data']

            # Apply modern cleaning
            cleaned_data = method_used(eeg_data)

            # Save cleaned data in the output directory with the same filename
            save_path = os.path.join(output_dir, filename)
            savemat(save_path, {'Clean_data': cleaned_data})

            print(f"✅ Saved cleaned file to {save_path}")

    print("🚀 All files processed successfully.")


if __name__ == "__main__":
    input_dir = '../Data/raw_data'
    method_name = 'SKLFast_ICA'
    filter_all_data(method_name,input_dir)
