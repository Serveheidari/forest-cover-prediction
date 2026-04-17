# Forest Cover Type Classification (Deep Learning)

This project uses deep learning to predict forest cover types based on cartographic features such as elevation, slope, soil type, and distance to hydrology.

---

## Dataset

- Source: US Forest Service (USFS)
- Observations: 581,012
- Features: 54 input features
- Target: 7 forest cover types

---

## Objective

To build a multi-class classification model that accurately predicts forest cover type using only cartographic variables.

---

## Model

- Framework: TensorFlow + Keras
- Architecture:
  - Dense (128) → ReLU
  - Dropout (0.3)
  - Dense (64) → ReLU
  - Dropout (0.3)
  - Dense (32) → ReLU
  - Output (7 classes, Softmax)
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam

---

## Preprocessing

- Feature scaling using StandardScaler
- Train / Validation / Test split (70/15/15)
- Target labels adjusted from (1–7) → (0–6)

---

## Results

- Accuracy: ~86%
- Good generalization (low overfitting)
- Strong performance on major classes

### Model Accuracy
![Accuracy](models/accuracy.png)

### Model Loss
![Loss](models/loss.png)

### Confusion Matrix
![Confusion Matrix](models/confusion_matrix.png)

---

##  Analysis

- High performance for dominant classes
- Misclassifications occur between similar forest types
- Lower recall for minority classes due to class imbalance

---

## Improvements

- Apply class weighting or SMOTE
- Try advanced models (XGBoost, CNN)
- Hyperparameter tuning (GridSearch)
- Feature engineering

---

## How to Run

```bash
pip install -r requirements.txt
python main.py

