# Inverse Feature Learning for Classification

This project is a simplified implementation of Inverse Feature Learning (IFL) based on error representation learning.

## Project Overview
The model uses K-Means clustering to generate centroid-based distance features, which act as error-based features. These new features are combined with the original dataset and used to train a Logistic Regression classifier.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Files
- `main.py` → main implementation
- `requirements.txt` → required libraries
- `dataset/iris.csv` → dataset
- `outputs/result.txt` → saved accuracy
- `outputs/accuracy.png` → confusion matrix

## How to Run
```bash
pip install -r requirements.txt
python main.py

Initial project upload
```
