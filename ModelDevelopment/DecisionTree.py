from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

sdss_data = pd.read_csv('SDSS_data.csv', low_memory=False)

sdss_data = sdss_data.reset_index()
sdss_data=sdss_data.iloc[1:]
sdss_data.columns=[ 'objid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun','camcol', 'field', 'specobjid', 'class', 'redshift', 'plate','mjd', 'fiberid']
columns_to_convert = [ 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun','camcol', 'field',  'redshift', 'plate','mjd', 'fiberid']

for column in columns_to_convert:
    sdss_data[column] = pd.to_numeric(sdss_data[column])
df=sdss_data[["ra","dec","u","g","r","i","z","redshift","class"]]

Y=df[['class']]
X=df.drop('class', axis=1)

#Transform Y into numeric values
le = LabelEncoder()
class_features=Y['class']
le.fit(class_features)
Y = le.transform(class_features)

#Scale X

# Assuming X is your feature matrix
# Create an instance of StandardScaler
scaler = StandardScaler()

# Fit the scaler to your data and transform it
X = scaler.fit_transform(X)


# Perform the train-test split
# By default, test_size is set to 0.25, meaning 75% of the data will be used for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,stratify=Y, random_state=42)
tree_model = DecisionTreeClassifier()

# Train the model on the training data
tree_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_test = tree_model.predict(X_test)
y_pred_train = tree_model.predict(X_train)


# Evaluate Train Model
accuracy_train = accuracy_score(y_train, y_pred_train)
classification_rep_train = classification_report(y_train, y_pred_train)
print(f'Train Accuracy: {accuracy_train:.4f}')
print('Train Classification Report:\n', classification_rep_train)
confusion_matrix_train = ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, display_labels=le.classes_)


# Evaluate Test Model
accuracy_test = accuracy_score(y_test, y_pred_test)
classification_rep_test = classification_report(y_test, y_pred_test)
print(f'Test Accuracy: {accuracy_test:.4f}')
print('Test Classification Report:\n', classification_rep_test)
confusion_matrix_test = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, display_labels=le.classes_)
