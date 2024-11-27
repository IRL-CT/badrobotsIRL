import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from create_data_splits import create_data_splits
import wandb

def log_metrics_and_features(file_name, feature_groups, X_train, X_test, y_train, y_test, grid_search, best_rf):
    with open(file_name, 'w') as file:
        file.write("Best GridSearch Parameters:\n")
        file.write(str(grid_search.best_params_) + "\n\n")

        file.write("Feature Importance:\n")
        importances = best_rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': features.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        file.write(feature_importance_df.to_string(index=False) + "\n\n")

        file.write("Metrics for Different Feature Groups:\n")
        for group_name, feature_subset in feature_groups.items():
            accuracy, f1, precision, recall = run_model_on_feature_subset(X_train, X_test, y_train, y_test, feature_subset, best_rf)
            
            file.write(f"{group_name}:\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write(f"F1 Score: {f1:.4f}\n")
            file.write(f"Precision: {precision:.4f}\n")
            file.write(f"Recall: {recall:.4f}\n\n")

            wandb.log({
                f"{group_name}_Accuracy": accuracy,
                f"{group_name}_F1_Score": f1,
                f"{group_name}_Precision": precision,
                f"{group_name}_Recall": recall,
            })

def run_model_on_feature_subset(X_train, X_test, y_train, y_test, feature_subset, best_rf):
    X_train_sub = X_train[feature_subset]
    X_test_sub = X_test[feature_subset]

    best_rf.fit(X_train_sub, y_train)
    y_pred = best_rf.predict(X_test_sub)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    return accuracy, f1, precision, recall

df = pd.read_csv("../preprocessing/merged_features/all_participants_merged_correct.csv")

participant_frames_labels = df.iloc[:, :4]
df_pose = df.iloc[:, 4:28]
df_facial = df.iloc[:, 28:63]
df_audio = df.iloc[:, 63:]

features = df.iloc[:, 4:]
target = df.iloc[:, 2].values.astype('int')

splits = create_data_splits(df, fold_no=0, num_folds=5, seed_value=42, sequence_length=5)
X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
}

wandb.init(project="rf_feature_importance", config=param_grid)

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

wandb.config.update(grid_search.best_params_)

feature_groups = {
    'All': features.columns.tolist(),
    'No Pose': [col for col in features.columns if col not in df_pose.columns],
    'No Facial': [col for col in features.columns if col not in df_facial.columns],
    'No Audio': [col for col in features.columns if col not in df_audio.columns],
}

log_metrics_and_features('model_results.txt', feature_groups, X_train, X_test, y_train, y_test, grid_search, best_rf)


# results_df = pd.DataFrame(results).T
# metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
# results_df[metrics].plot(kind='bar', figsize=(12, 6), title='Performance Comparison with Different Feature Exclusions')
# plt.ylabel('Score')
# plt.xticks(rotation=45)
# plt.show()