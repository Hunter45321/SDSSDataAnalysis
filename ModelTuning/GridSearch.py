from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from Preprocessing import *

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Adjust the parameter grid around the best parameters from Randomized CV
param_grid = {
    'n_estimators': [100],
    'min_samples_split': [5,10,20],
    'max_features': [2,4,6]
}

# Create Grid Search Cross-Validation object
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, scoring='f1_weighted', verbose=4)

# Fit the Grid Search to the data using the best parameters from Randomized CV
grid_search.fit(X_train, y_train)  # Replace X, y with your data


# Print the best parameters and the corresponding accuracy
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

# Make predictions on the test data
y_pred_test = grid_search.best_estimator_.predict(X_test)
y_pred_train = grid_search.best_estimator_.predict(X_train)

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