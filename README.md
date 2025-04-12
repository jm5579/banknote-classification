# banknote-classification

# Automated USD Banknote Authentication Using Convolutional Neural Networks

This project is a capstone assignment for the Deep Learning with Pytorch course.  
The objective is to build an automated system to classify and authenticate USD banknotes using deep learning techniques, specifically using Convolutional Neural Networks (CNN) implemented in PyTorch.

## Project Structure

- data/                  # Dataset (not uploaded to GitHub due to size)
- notebook.ipynb         # Main Jupyter Notebook with all code and outputs
- README.md              # Project documentation (this file)
- requirements.txt       # Project dependencies

## Dataset

- Name: USD Bill Classification Dataset
- Source: Kaggle
- Classes: 6 denominations (1, 2, 5, 10, 50, 100 dollars)
- Total Images: Approximately 3912 images
- Split:
  - Training: 2738 images
  - Validation: 586 images
  - Test: 588 images

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- Seaborn
- Scikit-learn

## Model Architecture

- Model: Pre-trained ResNet50
- Modifications:
  - Final layer adjusted for 6 output classes
  - Used transfer learning for faster convergence
- Hyperparameters:
  - Learning rate: 0.001
  - Optimizer: Adam
  - Epochs: 30
  - Early stopping patience: 20

## Training Results

- Training Accuracy: Approximately 99%
- Test Accuracy: Approximately 99%
- Class-wise Accuracy:
  - 1 Dollar: 98.08%
  - 10 Dollar: 100%
  - 100 Dollar: 98.73%
  - 2 Dollar: 98.91%
  - 5 Dollar: 98.84%
  - 50 Dollar: 100%

## Evaluation Metrics

- Precision, Recall, F1-Score
- Confusion Matrix
- Class-wise Accuracy
- Prediction Confidence Scores

## Project Outcomes

- Built an automated system for banknote authentication
- Achieved high accuracy, including with noisy images
- Visualized model performance during training
- Compared models (ResNet18 versus ResNet50) and improved results

## Lessons Learned

- Switched from ResNet18 to ResNet50 to improve feature extraction and performance
- Increased training epochs to 50 for stable convergence
- Used early stopping with a patience of 20 to prevent overfitting
- Applied data augmentation and tested robustness with noisy images
- Gained experience in building end-to-end deep learning pipelines

## Future Work

- Optimize the model for mobile and embedded deployment
- Use larger and more diverse datasets for better generalization
- Explore model quantization and pruning for faster inference
