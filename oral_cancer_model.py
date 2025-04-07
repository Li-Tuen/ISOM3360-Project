import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load dataset
data = pd.read_csv('oral_cancer_prediction_dataset.csv')

# Handle missing values
data = data.dropna()

# Separate features and target
X = data.drop(['Oral Cancer (Diagnosis)', 'Treatment Type', 'Survival Rate (5-Year, %)', 'Cost of Treatment (USD)', 'Economic Burden (Lost Workdays per Year)', 'Early Diagnosis'], axis=1)
y = data['Oral Cancer (Diagnosis)']

# Encode target variable
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

# Define numerical columns before pipeline
# Correct column names if necessary
num_cols = ['Age', 'Tumor Size (cm)']
remaining_cat_cols = ['Gender', 'Country', 'Tobacco Use', 'Alcohol Consumption', 'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene', 'Family History of Cancer', 'Compromised Immune System', 'Oral Lesions', 'Unexplained Bleeding', 'Difficulty Swallowing', 'White or Red Patches in Mouth']
diet_col = ['Diet (Fruits & Vegetables Intake)']

# Split dataset first
X_temp, X_eval, y_temp, y_eval = train_test_split(X, y, test_size=0.3, stratify=y)

# Add missing train-val split
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp)


# Create and apply preprocessing pipeline
preprocessor = make_pipeline(
    ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('diet', OrdinalEncoder(categories=[['Low', 'Moderate', 'High']]), diet_col),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), remaining_cat_cols)
    ], verbose_feature_names_out=False),
    SelectKBest(score_func=f_classif, k=10)
)

preprocessor.fit(X_train, y_train)
feature_names = preprocessor.named_steps['columntransformer'].get_feature_names_out()
selected_features = preprocessor.named_steps['selectkbest'].get_feature_names_out(feature_names)

X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_eval_processed = preprocessor.transform(X_eval)


# Save test set
eval_df = pd.concat([pd.DataFrame(X_eval_processed, columns=selected_features), 
                   pd.Series(y_eval, name='Oral Cancer (Diagnosis)')], axis=1)
eval_df.to_csv('test.csv', index=False)

# Build logistic regression neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02),
                         input_shape=(X_train_processed.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

# Define early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)
callbacks = [early_stopping, reduce_lr]

# Simplified training section with callbacks
history = model.fit(X_train_processed, y_train,
                    epochs=12,
                    batch_size=16,
                    validation_data=(X_val_processed, y_val),
                    verbose=0,
                    callbacks=callbacks)

# Evaluate final model
y_pred_final = model.predict(X_eval_processed)
y_pred_class_final = (y_pred_final > 0.5).astype(int)

print("\nFinal Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_eval, y_pred_class_final):.2f}")
print(f"Precision: {precision_score(y_eval, y_pred_class_final):.2f}")
print(f"Recall: {recall_score(y_eval, y_pred_class_final):.2f}")
print(f"ROC AUC: {roc_auc_score(y_eval, y_pred_final):.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_eval, y_pred_class_final))

# Plot training history
# Update training plot with validation metrics
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Training History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_history.png')
plt.show()

# Save model
model.save('oral_cancer_model.keras')