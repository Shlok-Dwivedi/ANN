# Lab 6: Artificial Neural Network on the Seeds Dataset

Multi-class classification of seeds into three classes using a feed-forward Artificial Neural Network built with TensorFlow/Keras.

## Overview

The notebook covers the full workflow: exploratory data analysis, preprocessing, building and training an ANN, and evaluating it on a held-out test set.

**Dataset:** `seeds_new.csv` has 210 samples, 7 numerical features (Area, Perimeter, Compactness, Length of kernel, Width of kernel, Asymmetry coefficient, Length of kernel groove) and a class label (1, 2, 3) with 70 samples per class. There are no missing values or duplicates.

## Workflow

1. Load the data, check for missing values and duplicates, and check the class balance
2. EDA: pie chart, histograms, boxplots, correlation heatmap, scatter plot and pair plot
3. Shift labels from 1, 2, 3 to 0, 1, 2 for Keras
4. 80:20 stratified train-test split (168 train, 42 test)
5. Standardise features with `StandardScaler` (fitted on training data only)
6. Build the ANN, compile it and train it with early stopping
7. Plot accuracy and loss curves, evaluate on the test set, and report metrics and a confusion matrix

## Model

| Layer | Units | Activation |
|-------|-------|------------|
| Input | 7 | n/a |
| Dense | 16 | ReLU |
| Dense | 8 | ReLU |
| Dense | 3 | Softmax |

- Total parameters: 291
- Optimizer: Adam (learning rate 0.001)
- Loss: sparse categorical crossentropy
- Epochs: up to 100, batch size 16, validation split 0.20
- Callback: `EarlyStopping` (monitor `val_loss`, patience 10, restore best weights)

## Results (from the notebook run)

| Metric | Value |
|--------|-------|
| Final training accuracy | 0.9478 |
| Final validation accuracy | 0.9118 |
| Test accuracy | 0.8810 |
| Test loss | 0.3630 |
| Weighted precision | 0.8995 |
| Weighted recall | 0.8810 |
| Weighted F1-score | 0.8731 |

All 5 test errors were class 1 samples (3 predicted as class 2, 2 as class 3). Classes 2 and 3 were predicted perfectly. Training ran all 100 epochs, so early stopping did not trigger.

## Project structure

```
lab6-ann-seeds/
├── ML_Lab6_updated.ipynb   # notebook
├── seeds_new.csv           # dataset (add this yourself)
├── requirements.txt
└── README.md
```

## How to run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ML_Lab6_updated.ipynb
```

Place `seeds_new.csv` in the same folder as the notebook, then run all cells.

## Notes

- Results can vary slightly between runs because the weights are initialised randomly. Only the data split is seeded (`random_state=42`).
- The notebook was run on Python 3.13 on Windows, where TensorFlow uses the CPU only. That is enough for a model this small.
