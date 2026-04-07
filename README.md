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
```

---

## 3. Open GitHub
Go to GitHub and sign in.

Then:
1. Click **New**
2. Repository name: `inverse-feature-learning-project`
3. Keep it **Public**
4. Click **Create repository**

---

## 4. Upload project using website
This is the easiest method.

After repo is created:
1. Click **uploading an existing file**
2. Drag and drop your whole project files there  
   or upload file-by-file:
   - `main.py`
   - `requirements.txt`
   - `README.md`
   - `dataset` folder files
   - `outputs` folder files

Then in the commit box write:

```text
Initial project upload
```
