# 🧠 EEG-Based Stress Level Classification Using CNNs  

## 📌 Project Overview  
This project explores the application of **Convolutional Neural Networks (CNNs)** for classifying stress levels based on **EEG data**. It includes data preprocessing, model training, and evaluation, aiming to improve classification accuracy through advanced neural network architectures and signal processing techniques.  

## 📚 Project Structure  
The repository is organized as follows:  

- **`Data/`** – Contains EEG datasets used for training and evaluation.  
- **`support_func/`** – Includes auxiliary Python scripts for data processing and model support:  
   - ***`dataset_class.py`*** – Defines the EEG dataset structure.  
   - ***`filters.py`*** – Implements filtering techniques for EEG signal preprocessing.  
   - ***`import_data.py`*** – Handles EEG dataset loading and formatting.  
   - ***`NN_classes.py`*** – Contains neural network architectures, including CNN-based models.  
- **`date_analysis.ipynb`** – Jupyter Notebook for dataset analysis and preprocessing.  
- **`neural_network.ipynb`** – Implements and trains the CNN model for EEG classification.  
- **`main.py`** – Main script for model evaluation and performance testing.  
- **`subject_head_with_electrodes.png`** – Visual representation of EEG electrode placements.  

## 📊 Dataset  
The dataset comprises EEG recordings labeled with corresponding stress levels.  
Detailed information about the dataset is available in the **`Data/`** directory.  

## 🔬 Models Implemented  
- **Convolutional Neural Network (CNN)** designed for EEG stress classification.  
- Baseline models to compare CNN performance with simpler architectures.  

## ⚙️ Installation & Requirements  
Ensure you have the necessary dependencies installed before running the scripts:  

```bash
pip install numpy pandas matplotlib torch torchvision scikit-learn mne scipy seaborn
```

## 🚀 How to Run  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/your-repo-link.git
   cd your-repo-name
   ```
2. **Preprocess the EEG dataset:**  
   ```bash
   jupyter notebook date_analysis.ipynb
   ```
3. **Train the CNN model:**  
   ```bash
   jupyter notebook neural_network.ipynb
   ```
4. **Evaluate the model:**  
   ```bash
   python main.py
   ```

## 📈 Results & Performance Analysis  
The CNN model achieved **poor accuracy** on the test dataset, highlighting challenges in classifying stress levels from EEG signals. Future work may explore alternative architectures, feature extraction methods, and improved preprocessing techniques.  

## 🐝 References  
- [Assignment: EEG-Based Emotion Recognition Using Neural Networks](Assignment_%20EEG-Based%20Emotion%20Recognition%20Using%20Neural%20Networks.pdf)  
- [Presentation: Assignment n1](Assignment%20n1.pptx)  
