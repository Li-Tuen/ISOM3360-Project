import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
df = pd.read_csv('oral_cancer_prediction_dataset.csv')

label = 'Survival Rate (5-Year, %)'
categorical_columns = ['Cancer Stage', 'Treatment Type']
numerical_columns = ['Cost of Treatment (USD)', 'Tumor Size (cm)', 'Economic Burden (Lost Workdays per Year)']

X = df.drop(['ID', label], axis=1)
y = df[label]
# Create labels with special case for 100%
y_encoded = np.zeros(len(y), dtype=int)
y_encoded[(y >= 0) & (y < 30)] = 0
y_encoded[(y >= 30) & (y < 50)] = 1
y_encoded[(y >= 50) & (y < 80)] = 2
y_encoded[(y >= 80) & (y < 100)] = 3
y_encoded[y == 100] = 4

# Print out distribution of the target variable
print("Distribution of the target variable:")
print(pd.Series(y_encoded).value_counts(),'\n')

# Split the data with stratification
X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp)

# Save testing set to CSV
# Convert y_test from numpy array to pandas Series before concatenation
y_test_series = pd.Series(y_test, name=label)
test_data = pd.concat([X_test, y_test_series], axis=1)
test_data.to_csv('test_set.csv', index=False)


# Define the preprocessor with different scaling for Age and other numerical features
# and ordinal encoding for Diet column
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['lbfgs', 'saga'],
    'classifier__max_iter': [1000],
    'classifier__class_weight': ['balanced', None]
}

# Create a pipeline with SMOTE for handling class imbalance
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42))
])

# Set up cross-validation with stratification
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1
)

# Train the model
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Make predictions on validation set
y_val_pred = best_model.predict(X_val)
y_val_proba = best_model.predict_proba(X_val)

# Evaluate the model
print("\nValidation Set Performance:")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['<30%', '30-50%', '50-80%', '80-100%', '100%'],
            yticklabels=['<30%', '30-50%', '50-80%', '80-100%', '100%'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Plot ROC curve for multiclass
plt.figure(figsize=(10, 8))
for i in range(5):
    precision, recall, _ = precision_recall_curve(y_val == i, y_val_proba[:, i])
    avg_precision = average_precision_score(y_val == i, y_val_proba[:, i])
    plt.plot(recall, precision, lw=2, 
             label=f'Class {i} (AP = {avg_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc='best')
plt.savefig('precision_recall_curve.png')
plt.close()

# Save the model
joblib.dump(best_model, 'survival_rate_model.pkl')
print("\nModel saved as 'survival_rate_model.pkl'")

# Make predictions on test set
y_test_pred = best_model.predict(X_test)
print("\nTest Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


# Apply the preprocessor to the entire feature set X
X_processed = preprocessor.fit_transform(X)

# Get feature names after preprocessing
# Handle OneHotEncoder names
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
# Combine all feature names in the correct order
feature_names = numerical_columns + list(cat_feature_names)

# Convert the processed data pandas DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

# Combine processed features with the encoded target variable
df_processed_corr = pd.concat([X_processed_df, pd.Series(y_encoded, name='Survival_Rate_Encoded', index=X.index)], axis=1)

# Calculate the correlation matrix
corr_matrix_full = df_processed_corr.corr()

# Plot the full correlation heatmap
plt.figure(figsize=(20, 18))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix_full, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
            linewidths=.5, cbar_kws={"shrink": .5}, annot=False)

plt.title('Correlation Heatmap of All Processed Features and Encoded Survival Rate', fontsize=16)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('correlation_heatmap_full.png')
plt.close()
print("\nFull correlation heatmap saved as 'correlation_heatmap_full.png'")

