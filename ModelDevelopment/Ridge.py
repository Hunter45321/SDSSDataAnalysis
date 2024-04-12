from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from Preprocessing import *

ridge_model = linear_model.RidgeClassifier()

# Train the model on the training data
ridge_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_test = ridge_model.predict(X_test)
y_pred_train = ridge_model.predict(X_train)

# Evaluate Train Model
accuracy_train = accuracy_score(y_train, y_pred_train)
classification_rep_train = classification_report(y_train, y_pred_train)
print(f'Train Accuracy: {accuracy_train:.2f}')
print('Train Classification Report:\n', classification_rep_train)
confusion_matrix_train = ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, display_labels=le.classes_)

# Evaluate Test Model
accuracy_test = accuracy_score(y_test, y_pred_test)
classification_rep_test = classification_report(y_test, y_pred_test)
print(f'Test Accuracy: {accuracy_test:.2f}')
print('Test Classification Report:\n', classification_rep_test)
confusion_matrix_test = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, display_labels=le.classes_)