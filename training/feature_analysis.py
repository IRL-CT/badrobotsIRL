import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from create_data_splits import create_data_splits

df = pd.read_csv("../preprocessing/merged_features/all_participants_merged_correct.csv")

participant_frames_labels = df.iloc[:, :4]
df_pose = df.iloc[:, 4:28]
df_facial = df.iloc[:, 28:63]
df_audio = df.iloc[:, 63:]

features = df.iloc[:, 4:]
target = df.iloc[:, 2].values.astype('int')

splits = create_data_splits(df, fold_no=0, num_folds=5, seed_value=42, sequence_length=30)

X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy of optimal Random Forest: {accuracy:.4f}")
print(f"Macro F1 score: {f1:.4f}")

importances = best_rf.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df.head(10))

feature_groups = {
    'All': features.columns.tolist(),
    'No Pose': [col for col in features.columns if col not in df_pose.columns],
    'No Facial': [col for col in features.columns if col not in df_facial.columns],
    'No Audio': [col for col in features.columns if col not in df_audio.columns],
}

def run_model_on_feature_subset(X_train, X_test, y_train, y_test, feature_subset):
    X_train_sub = X_train[feature_subset]
    X_test_sub = X_test[feature_subset]

    best_rf.set_params(**grid_search.best_params_)
    best_rf.fit(X_train_sub, y_train)

    y_pred = best_rf.predict(X_test_sub)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    return accuracy, f1, precision, recall

results = {}
for group_name, features in feature_groups.items():
    accuracy, f1, precision, recall, auc_roc, cm = run_model_on_feature_subset(X_train, X_test, y_train, y_test, features)
    results[group_name] = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'AUC-ROC': auc_roc,
        'Confusion Matrix': cm
    }

results_df = pd.DataFrame(results).T
print(results_df)

metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC-ROC']
results_df[metrics].plot(kind='bar', figsize=(12, 6), title='Performance Comparison with Different Feature Exclusions')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.show()