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

titanic = pd.read_csv('titanic.csv')
titanic = titanic.drop(columns=["PassengerId", "Name", "SibSp", "Ticket", "Cabin", "Embarked", "Sex"])
titanic = titanic.drop_duplicates()
titanic = titanic.dropna()


# You will now build a simple Logistic Regression model using the sklearn library
# You should use the following columns as your features Pclass, Age, Parch, Fare for your X values
# The Y value should be the survived column

survived = titanic.pop("Survived")
X_train, X_test, y_train, y_test = train_test_split(titanic, survived)
model = LogisticRegression().fit(X=X_train, y=y_train)
report = classification_report(y_train, model.predict(X_train))
c_matrix = confusion_matrix(y_test, model.predict(X_test))
# probs = model.predict_proba(X_test)
# preds = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
plt.show()



# Check the score of your model for the train values
# Check the score of your model for the test values
# Produce a classification report for your y values

# Produce a confusion matrix for your model
# Plot a ROC curve for your model