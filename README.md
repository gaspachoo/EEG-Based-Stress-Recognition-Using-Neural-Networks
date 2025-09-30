# 🧠 EEG-Based Stress Level Classification Using CNNs  

## 📌 Project Overview  
This project explores the application of **Convolutional Neural Networks (CNNs)** for classifying stress levels based on **EEG data**. It includes data preprocessing, model training, and evaluation, aiming to improve classification accuracy through advanced neural network architectures and signal processing techniques.  

## 📚 Project Structure  
The repository is organized as follows:  

- **`Data/`** – Contains EEG datasets used for training and evaluation, including those downloaded and those generated using `filter_all_data.py` 
- **`support_func/`** – Includes auxiliary Python scripts for data processing and model support:
   - **`cleaning_algos.py`** – Contains mutliple data processing algorithms.
   - **`dataset_class.py`** – Defines the EEG dataset structure.
   - **`early_stopping_class.py`** – Defines the Early Stopping class.
   - **`filter_all_data.py`** – Implements filtering techniques for EEG signal preprocessing.  
   - **`import_data.py`** – Defines the functions to import data, labels and split behind train/val datasets.  
   - **`model_processing_cla.py`** and **`model_processing_reg.py`** – Handles train generating, loading, training, under and over-sampling for classification and regression.
   - **`NN_classes.py`** – Contains neural network architectures, including CNN-based models.
   - **`results_evaluation.py`** – Implements functions to show the results and plot confusion matrix.  
- **`date_analysis.ipynb`** – Jupyter Notebook for exploring the dataset and displaying multiple plots.
- **`main_cla.py`** and **`main_reg.py`** – Main scripts for model evaluation and performance testing (classification or regression)  
- **`subject_head_with_electrodes.png`** – Visual representation of EEG electrode placements.  

## 📊 Dataset  
The dataset comprises EEG recordings labeled with corresponding stress levels.  
In the **`Data/`** directory, you can find the downloaded data `artifcat_removal`, `raw_data`, `filtered_data`, `Coordinates.locs`, `scales.xls` and my own folders, named `[filtering_method]_filtered_data`.

## 🔬 Models Implemented 
- **SimpleNN**: A basic fully connected neural network that flattens EEG input, applies two hidden layers with ReLU activation, and outputs raw scores for classification.  
- **EEG_CNN**: A 1D CNN model with three convolutional layers, batch normalization, and max-pooling, followed by fully connected layers for EEG feature extraction and classification.  
- **SimpleNN2**: A refined version of SimpleNN with batch normalization and dropout for regularization, reducing overfitting while maintaining a simple architecture.  
- **EEG_CNN2**: A lighter CNN model with fewer filters and dropout layers, balancing complexity and generalization for EEG-based classification.  
- **EEG_CNN_GRU**: A hybrid CNN-GRU model that extracts spatial EEG features using CNN and captures temporal dependencies with a GRU layer before classification.  

## ⚙️ Installation & Requirements  
Install the required Python packages before running the scripts:

```bash
pip install -r requirements.txt
```

## How to Run  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/your-repo-link.git
   cd your-repo-name
   ```
2. **(Facultative) Explore the dataset:**  
   ```bash
   jupyter notebook data_analysis.ipynb
   ```

3. **(Facultative) Filter the EEG dataset:**  
   ```bash
   python filter_all_data.py
   ```
4. **Train and evaluate a model:**  
   ```bash
   python main.py
   ```


## 📈 Results & Performance Analysis  
The CNN model achieved **poor accuracy** on the test dataset, highlighting challenges in classifying stress levels from EEG signals. Future work may explore alternative architectures, feature extraction methods, and improved preprocessing techniques.  

## 🐝 References  
- [Assignment: EEG-Based Emotion Recognition Using Neural Networks](Assignment_%20EEG-Based%20Emotion%20Recognition%20Using%20Neural%20Networks.pdf)  
- [EEG-Based Emotion Recognition Using Neural Networks (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2352340921010465)  
- [SAM 40 Dataset (Figshare)](https://figshare.com/articles/dataset/SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_Mirror_Image_Recognition_Task/14562090/1)
- [Introduction to EEG (AlexEnge)](https://alexenge.github.io/intro-to-eeg/misc/index.html)

## License
This project is licensed under the [MIT License](LICENSE).

