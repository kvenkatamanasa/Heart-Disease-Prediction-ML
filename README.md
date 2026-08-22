# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project is a **Heart Disease Prediction System** built using **Python and Machine Learning**.

The objective of this project is to analyze patient health information and predict whether a person is likely to have heart disease.

Several machine learning classification algorithms are trained and evaluated, including:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Gradient Boosting Classifier

The final model used for prediction is **Gradient Boosting Classifier**.

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze the heart disease dataset.
* Perform basic data exploration.
* Check for missing values.
* Separate input features and target variable.
* Split the dataset into training and testing data.
* Train multiple machine learning classification models.
* Compare model predictions.
* Train the final Gradient Boosting model.
* Save and load the trained machine learning model.
* Predict heart disease for new patient data.
* Visualize important patterns in the dataset.

---

## 🛠️ Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Programming Language           |
| Pandas           | Data manipulation and analysis |
| Scikit-learn     | Machine Learning               |
| Matplotlib       | Data visualization             |
| Seaborn          | Statistical visualization      |
| Joblib           | Saving and loading ML models   |
| Jupyter Notebook | Development environment        |

---

## 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── heart.csv
├── heart_disease_prediction.ipynb
├── requirements.txt
└── README.md
```
# 📊 Dataset

The project uses a dataset named:

```text
heart.csv
```

The dataset contains medical attributes of patients that can be used to predict heart disease.

### Important Features

| Feature    | Description                          |
| ---------- | ------------------------------------ |
| `age`      | Age of the patient                   |
| `sex`      | Gender                               |
| `cp`       | Chest pain type                      |
| `trestbps` | Resting blood pressure               |
| `chol`     | Cholesterol level                    |
| `fbs`      | Fasting blood sugar                  |
| `restecg`  | Resting electrocardiographic results |
| `thalach`  | Maximum heart rate achieved          |
| `exang`    | Exercise-induced angina              |
| `oldpeak`  | ST depression                        |
| `slope`    | Slope of peak exercise ST segment    |
| `ca`       | Number of major vessels              |
| `thal`     | Thalassemia                          |
| `target`   | Heart disease prediction target      |

---

# 🔍 Target Variable

The target column is:

```python
target
```

It contains two classes:

```text
0 → No Heart Disease
1 → Heart Disease
```

The features are separated from the target using:

```python
X = data.drop("target", axis=1)
Y = data["target"]
```

Where:

* `X` = Input features
* `Y` = Target variable

---

# 📥 Installation

Install the required Python libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

If you are using Jupyter Notebook:

```python
!pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

# 📚 Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import joblib
```

---

# 📖 Load Dataset

The dataset is loaded using Pandas:

```python
data = pd.read_csv("heart.csv")
```

View the first five records:

```python
data.head()
```

View the last five records:

```python
data.tail()
```

---

# 🔎 Data Exploration

Statistical information can be viewed using:

```python
data.describe()
```

This provides information such as:

* Count
* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

---

# ❌ Missing Value Check

Missing values are checked using:

```python
data.isnull().sum()
```

This helps determine whether any column contains missing data.

---

# 🧹 Feature Selection

The target column is separated from the input features:

```python
X = data.drop("target", axis=1)
Y = data["target"]
```

Here:

```text
X → Patient medical features
Y → Heart disease target
```

---

# ✂️ Train-Test Split

The dataset is divided into training and testing sets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
```

### Split Explanation

```text
80% → Training data
20% → Testing data
```

`random_state=42` ensures that the same split can be reproduced.

---

# 🤖 Machine Learning Models

Five classification algorithms are used.

## 1. Logistic Regression

```python
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train)
```

Logistic Regression is commonly used for binary classification problems.

---

## 2. K-Nearest Neighbors

```python
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
```

KNN predicts a class based on nearby observations.

---

## 3. Decision Tree

```python
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
```

Decision Tree uses a tree-like structure to make predictions.

---

## 4. Random Forest

```python
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
```

Random Forest combines multiple decision trees to improve prediction performance.

---

## 5. Gradient Boosting

```python
gb = GradientBoostingClassifier()
gb.fit(X_train, Y_train)
```

Gradient Boosting builds models sequentially to improve prediction performance.

---

# 🔮 Model Predictions

Predictions are generated using:

```python
Y_pred1 = lr.predict(X_test)
Y_pred2 = knn.predict(X_test)
Y_pred3 = dt.predict(X_test)
Y_pred4 = rf.predict(X_test)
Y_pred5 = gb.predict(X_test)
```

Each model produces a prediction for the test dataset.

---

# 📈 Model Evaluation

The current notebook uses:

```python
metrics.r2_score()
```

However, because this is a **classification problem**, classification metrics such as **accuracy, precision, recall, F1-score, and confusion matrix** are more appropriate than R².

A better evaluation approach would be:

```python
from sklearn.metrics import accuracy_score

print("Logistic Regression:",
      accuracy_score(Y_test, Y_pred1))

print("KNN:",
      accuracy_score(Y_test, Y_pred2))

print("Decision Tree:",
      accuracy_score(Y_test, Y_pred3))

print("Random Forest:",
      accuracy_score(Y_test, Y_pred4))

print("Gradient Boosting:",
      accuracy_score(Y_test, Y_pred5))
```

---

# 🏆 Final Model

The final model selected in the project is:

```python
GradientBoostingClassifier()
```

The model is trained using the complete dataset:

```python
gb = GradientBoostingClassifier()
gb_final = gb.fit(X, Y)
```

---

# 💾 Save the Machine Learning Model

The trained model can be saved using Joblib:

```python
joblib.dump(
    gb_final,
    "models/heart_disease_model.pkl"
)
```

This allows the trained model to be reused without training it again.

---

# 📂 Load the Saved Model

The saved model can be loaded using:

```python
model = joblib.load(
    "models/heart_disease_model.pkl"
)
```

---

# 👤 New Patient Prediction

A new patient's medical information can be provided as a Pandas DataFrame:

```python
data_new = pd.DataFrame({
    'age': 55,
    'sex': 1,
    'cp': 0,
    'trestbps': 140,
    'chol': 250,
    'fbs': 0,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.0,
    'slope': 1,
    'ca': 0,
    'thal': 2
}, index=[0])
```

The trained model predicts the result:

```python
prediction = model.predict(data_new)
```

---

# 🩺 Prediction Result

The prediction is interpreted using:

```python
if prediction[0] == 1:
    print("The person has Heart Disease")
else:
    print("The person does NOT have Heart Disease")
```

The model output is:

```text
1 → Heart Disease
0 → No Heart Disease
```

# 📊 Data Visualizations

Several visualizations are created to understand the dataset.

## 1. Age vs Target

A scatter plot is used to visualize the relationship between age and heart disease:

```python
plt.scatter(data["age"], data["target"], alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Target")
plt.title("Age vs Heart Disease")
plt.show()
```

---

## 2. Heart Disease Count by Age

A count plot shows the number of patients with and without heart disease across different ages.

```python
sns.countplot(
    data=data,
    x="age",
    hue="target"
)
```

---

## 3. Age Distribution by Target

A swarm plot is used to visualize age distribution for each target class.

```python
sns.swarmplot(
    data=data,
    x="target",
    y="age"
)
```

---

## 4. Age vs Cholesterol

A scatter plot displays the relationship between age and cholesterol.

```python
sns.scatterplot(
    data=data,
    x="age",
    y="chol",
    hue="target"
)
```

---

## 5. Heart Disease Classification

A stem plot displays the number of patients in each target category.

```text
No Disease → 0
Disease     → 1
```

---

## 6. Heart Disease Distribution

A pie chart shows the percentage distribution between the two target classes.

```python
data["target"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
```

---

## 7. Age Distribution

A histogram is used to understand the distribution of patient ages.

```python
sns.histplot(
    data["age"],
    bins=10,
    kde=True
)
```

---

# 🔄 Machine Learning Workflow

```text
              Heart Disease Dataset
                       │
                       ▼
                Data Collection
                       │
                       ▼
                 Data Loading
                       │
                       ▼
                Data Exploration
                       │
                       ▼
              Missing Value Checking
                       │
                       ▼
               Feature Selection
                       │
                       ▼
                Train/Test Split
                       │
                       ▼
              ┌───────────────────┐
              │ Machine Learning  │
              │      Models       │
              └───────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Logistic          KNN        Decision Tree
  Regression
        │              │              │
        └──────────────┼──────────────┘
                       │
                 Random Forest
                       │
                       ▼
             Gradient Boosting
                       │
                       ▼
                Final Model
                       │
                       ▼
              New Patient Data
                       │
                       ▼
                  Prediction
                       │
              ┌────────┴────────┐
              ▼                 ▼
        No Heart Disease   Heart Disease
```

---

# 🚀 How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/Heart-Disease-Prediction.git
```

### Step 2: Open the project

```bash
cd Heart-Disease-Prediction
```

### Step 3: Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 4: Make sure the dataset exists

Place:

```text
heart.csv
```

in the appropriate project directory.

### Step 5: Open the Jupyter Notebook

```bash
jupyter notebook
```

### Step 6: Run the notebook

Open:

```text
heart_disease_prediction.ipynb
```

and execute the cells from top to bottom.

---

# 📌 Project Features

* ✅ Heart disease dataset analysis
* ✅ Data exploration
* ✅ Missing value detection
* ✅ Feature and target separation
* ✅ Train-test splitting
* ✅ Multiple ML classification algorithms
* ✅ Model prediction
* ✅ Model saving using Joblib
* ✅ Model loading
* ✅ New patient prediction
* ✅ Data visualization
* ✅ Heart disease classification

---

# 🔮 Future Improvements

The project can be improved by adding:

* Model accuracy comparison
* Confusion matrix
* Classification report
* Precision, recall and F1-score
* ROC-AUC curve
* Feature importance
* Hyperparameter tuning
* Cross-validation
* StandardScaler for applicable models
* A Streamlit web application
* User-friendly prediction form
* Deployment using cloud platforms

---

# ⚠️ Important Note

This project is created for **learning and educational purposes**.

The machine learning prediction should **not be considered a medical diagnosis**. Real medical decisions should always be made by qualified healthcare professionals.

---

# 👩‍💻 Author

**Kammineni Venkata Manasa**

