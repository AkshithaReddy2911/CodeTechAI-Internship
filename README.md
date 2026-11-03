# CodeTechAI-Internship
Problem: Classify chest X-ray images as NORMAL or PNEUMONIA using deep learning.

Dataset: Chest X-Ray Images (Pneumonia) from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Note on the dataset: The dataset is very large (about 2 GB), so it could not be attached to this repository (GitHub does not allow files over 100 MB). It can be downloaded from the Kaggle link above.

Approach:

Transfer learning with MobileNetV2 (pre-trained on ImageNet)
Images resized to 224x224, with data augmentation (flip, rotation, zoom)
15% of the training images used for validation; the official test folder used only for final evaluation
Class weights to handle the imbalance between NORMAL and PNEUMONIA images
Two-phase training: top layers first, then fine-tuning the last layers with a small learning rate
Evaluation with accuracy, precision, recall, F1, confusion matrix and ROC-AUC
Results: Because of the large size of the dataset (several GB) and limited computing resources, the model could not be trained for enough epochs, so fully accurate final results could not be obtained. The complete code and pipeline are in the notebook, and with more compute time the notebook can be run end to end to get the final accuracy, confusion matrix and ROC-AUC.

Conclusion: The full pipeline (data loading, augmentation, transfer learning, fine-tuning and evaluation) has been implemented. Final accuracy depends on training on the full dataset with enough compute.

Tools used: Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn

File: task1_medical_image_classification.ipynb
