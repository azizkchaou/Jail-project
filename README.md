# Criminal Sentencing Prediction — Neural Network from Scratch

A multi-class classification neural network built from scratch using **NumPy only** to predict criminal sentencing outcomes across 7 supervision levels.

---

## 📌 Project Overview

This project aims to predict the type of criminal sentence assigned to an individual based on structured case features. The dataset was sourced from an **official U.S. government website**, making it a real-world dataset with all the challenges that come with it — noise, class imbalance, and limited signal.

---

## 📂 Dataset

- **Source:** Official U.S. government criminal justice database
- **Size:** ~296,000 records
- **Task:** Multi-class classification (7 classes)
- **Features:** 2,271 input features (sparse, TF-IDF style)

### Target Classes

| Class | Label | Samples |
|-------|-------|---------|
| 0 | Intensive Community Supervision | 11,259 |
| 1 | Deferred Conditional Programs | 10,328 |
| 2 | County Jail Intensive | 11,700 |
| 3 | Maximum Restriction | 781 |
| 4 | Minimal Restriction | 3,553 |
| 5 | Standard Probation | 100,106 |
| 6 | State Prison | 159,001 |

> ⚠️ Classes 5 and 6 account for ~88% of the data — severe class imbalance.

---

## 🧠 Model Architecture

```
Input (2271)
    ↓
Dense Layer (512) + Leaky ReLU + Dropout(0.8)
    ↓
Dense Layer (256) + Leaky ReLU + Dropout(0.8)
    ↓
Dense Layer (64)  + Leaky ReLU + Dropout(0.8)
    ↓
Output (7) + Softmax
```

---

## ⚙️ Implementation Details

All components were implemented from scratch using NumPy:

| Component | Details |
|-----------|---------|
| Weight Initialization | He (hidden layers), Xavier (output layer) |
| Activation Functions | Leaky ReLU (hidden), Softmax (output) |
| Loss Function | Focal Loss (γ=2.0) |
| Optimizer | Adam (β1=0.9, β2=0.999) |
| Regularization | Dropout (keep_prob=0.8) |
| Input Scaling | MaxAbsScaler (sparse-compatible) |
| Batch Size | 512 |
| Learning Rate | 0.001 |

---

## 📊 Results

```
              precision    recall  f1-score   support

           0       0.58      0.37      0.45      2388
           1       0.35      0.11      0.17      2056
           2       0.33      0.05      0.09      2242
           3       0.82      0.48      0.60       162
           4       0.30      0.13      0.18       717
           5       0.56      0.48      0.52     20044
           6       0.68      0.84      0.75     31737

    accuracy                           0.63     59346
   macro avg       0.52      0.35      0.40     59346
weighted avg       0.61      0.63      0.61     59346
```

---

## 🔍 Key Challenges

**1. Class Imbalance**
The two majority classes dominated training, causing the model to ignore minority classes. Addressed using Focal Loss which down-weights easy examples and focuses learning on hard ones.

**2. Real-World Data Quality**
Being sourced from a government database, the data contained noise and missing signal that no model architecture can fully overcome. XGBoost was also tested and hit a similar accuracy ceiling (~63%).

**3. Gradient Instability**
With 4 hidden layers and sparse inputs, gradient explosion was a recurring issue. Resolved by switching from SGD to Adam and tuning the gradient clipping threshold.

**4. Silent Bugs**
A misplaced indentation caused the parameter update to run only once per epoch instead of once per batch — effectively wasting 99% of gradient computations. This was one of the most impactful bugs fixed during the project.

---

## 🛠️ Project Structure

```
├── neural_network.ipynb     # Main notebook with all implementation
├── README.md                # This file
```

---

## 📦 Dependencies

```
numpy
scikit-learn
scipy (for sparse matrix support)
```

---

## 🚀 How to Run

```python
# 1. Scale input data
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Initialize parameters
layer_dims = [input_features, 512, 256, 64, output_classes]
parameters = init_parameters(layer_dims)

# 3. Train
trained_parameters, costs = training(
    X_train_scaled, Yc_train, parameters,
    learning_rate=0.001, num_iterations=120,
    alpha=0.01, output_classes=7,
    X_test_t=X_test_t, Yc_test=Yc_test,
    batch_size=512
)
```

---

## 💡 Lessons Learned

- Always normalize sparse inputs before feeding into a deep network
- Weighted loss and Adam optimizer can conflict — use Focal Loss instead
- Balanced batching and weighted loss together double-count the imbalance correction and destabilize training
- Real-world data often has a performance ceiling that no model can break — understanding the data's limitations is itself a valuable outcome
- Building from scratch forces a deeper understanding of every component than using a framework

---

## 📋 Conclusion

The 63% accuracy ceiling was consistent across both the custom neural network and XGBoost, suggesting the limitation lies in the data itself rather than the model. Predicting criminal sentencing involves human judgment, legal nuance, and contextual factors that may not be fully captured in structured records.

---

*Built with NumPy only — no PyTorch, no TensorFlow.*
