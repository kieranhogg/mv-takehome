import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# You should drop any duplicates values
# You should drop any null values
# You should drop PassengerId, Name, SibSp, Ticket, Cabin, Embarked, Sex columns

df = pd.read_csv('titanic.csv')
columns__to_drop = ["PassengerId", "Name", "SibSp", "Ticket", "Cabin", "Embarked", "Sex"]
df = df.drop(columns=columns__to_drop)
df = df.drop_duplicates()
df = df.dropna(how='any')

# You will now build a simple Logistic Regression model using the sklearn library
# You should use the following columns as your features Pclass, Age, Parch, Fare for your X values
# The Y value should be the survived column

survived = df.pop("Survived")
X_train, X_test, y_train, y_test  = train_test_split(df, survived)
model = LogisticRegression().fit(X=X_train, y=y_train)

# Check the score of your model for the train values
print(model.score(X=X_train, y=y_train))

# Check the score of your model for the test values
print(model.score(X=X_test, y=y_test))

# Produce a classification report for your y values
report = classification_report(y_train, model.predict(X_train))

# Produce a confusion matrix for your model
c_matrix = confusion_matrix(y_test, model.predict(X_test))
print(f"TN: {c_matrix[0][0]} FN: {c_matrix[0][1]} TP: {c_matrix[1][0]} FP: {c_matrix[1][1]}")

# Plot a ROC curve for your model
y_scores = model.predict_proba(X_test)
y_scores = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()






