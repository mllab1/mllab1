import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\xyz\\OneDrive\\Desktop\\pima-indians-diabetes.csv")
df.head()

x = df.iloc[: , : -1]
y = df.iloc[: , -1]
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score

x_train , x_test ,y_train,y_test = train_test_split(x,y)
dt = DecisionTreeClassifier()
rf  = RandomForestClassifier(n_estimators = 30)
dt.fit(x_train,y_train)

rf.fit(x_train,y_train)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Predictions on the training set for Decision Tree
dt_train_predictions = dt.predict(x_train)
# Predictions on the testing set for Decision Tree
dt_test_predictions = dt.predict(x_test)

# Predictions on the training set for Random Forest
rf_train_predictions = rf.predict(x_train)
# Predictions on the testing set for Random Forest
rf_test_predictions = rf.predict(x_test)

# Evaluate Decision Tree on training set
dt_train_accuracy = accuracy_score(y_train, dt_train_predictions)
dt_train_precision = precision_score(y_train, dt_train_predictions)
dt_train_f1 = f1_score(y_train, dt_train_predictions)

# Evaluate Decision Tree on testing set
dt_test_accuracy = accuracy_score(y_test, dt_test_predictions)
dt_test_precision = precision_score(y_test, dt_test_predictions)
dt_test_f1 = f1_score(y_test, dt_test_predictions)

# Evaluate Random Forest on training set
rf_train_accuracy = accuracy_score(y_train, rf_train_predictions)
rf_train_precision = precision_score(y_train, rf_train_predictions)
rf_train_f1 = f1_score(y_train, rf_train_predictions)

# Evaluate Random Forest on testing set
rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)
rf_test_precision = precision_score(y_test, rf_test_predictions)
rf_test_f1 = f1_score(y_test, rf_test_predictions)

# Print the results
print("Decision Tree - Training Set:")
print(f"Accuracy: {dt_train_accuracy:.4f}")
print(f"Precision: {dt_train_precision:.4f}")
print(f"F1 Score: {dt_train_f1:.4f}")

print("\nDecision Tree - Testing Set:")
print(f"Accuracy: {dt_test_accuracy:.4f}")
print(f"Precision: {dt_test_precision:.4f}")
print(f"F1 Score: {dt_test_f1:.4f}")

print("\nRandom Forest - Training Set:")
print(f"Accuracy: {rf_train_accuracy:.4f}")
print(f"Precision: {rf_train_precision:.4f}")
print(f"F1 Score: {rf_train_f1:.4f}")

print("\nRandom Forest - Testing Set:")
print(f"Accuracy: {rf_test_accuracy:.4f}")
print(f"Precision: {rf_test_precision:.4f}")
print(f"F1 Score: {rf_test_f1:.4f}")
