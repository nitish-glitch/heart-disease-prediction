"""
train_model.py
==============
Heart Disease Prediction - Model Training, Cross-Validation & Hyperparameter Tuning Pipeline
AI Subject College Project

Improvements:
1. Stratified 5-Fold Cross-Validation for reliable, variance-reduced performance estimates.
2. Hyperparameter optimization using GridSearchCV for all 3 models:
   - Logistic Regression (Regularization strength C, solver)
   - Random Forest (n_estimators, max_depth, min_samples_split)
   - K-Nearest Neighbors (n_neighbors, weights, distance metric)
3. Direct before-vs-after accuracy and metric comparison.
4. Stratified 80/20 train/test split evaluation with confusion matrix.
5. Serialization of the best model (model.pkl), standardizer (scaler.pkl), and metadata (model_meta.json).
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(filepath: str = "heart_disease.csv"):
    """
    Load Cleveland Heart Disease dataset and validate row/column counts.
    """
    print("=" * 78)
    print("STEP 1: INGESTING DATASET")
    print("=" * 78)
    df = pd.read_csv(filepath)
    print(f"Loaded {filepath}: {df.shape[0]} patient records across {df.shape[1]} columns.")
    
    target_dist = df['target'].value_counts()
    print(f"Target Distribution: Class 0 (No Disease): {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%), "
          f"Class 1 (Has Disease): {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")
    return df

def run_5fold_cross_validation_and_tuning(X, y):
    """
    Evaluates baseline models using 5-Fold Stratified Cross-Validation,
    then executes GridSearchCV to tune hyperparameters for optimal accuracy.
    """
    print("\n" + "=" * 78)
    print("STEP 2: 5-FOLD STRATIFIED CROSS-VALIDATION & HYPERPARAMETER TUNING")
    print("=" * 78)
    
    # 5-Fold Stratified K-Fold preserves class balance across all 5 folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Baseline models
    baseline_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }
    
    print("\n>>> Baseline 5-Fold Cross-Validation Scores (Before Tuning):")
    baseline_cv_scores = {}
    for name, model in baseline_models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        baseline_cv_scores[name] = scores.mean()
        print(f"  {name:<22}: Mean CV Accuracy = {scores.mean()*100:.2f}% (Std: ±{scores.std()*100:.2f}%)")
        
    # Hyperparameter Tuning with GridSearchCV
    print("\n>>> Running Hyperparameter Optimization (GridSearchCV)...")
    
    # 1. Logistic Regression Grid
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid={
            'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            'solver': ['liblinear', 'lbfgs']
        },
        cv=cv,
        scoring='accuracy'
    )
    lr_grid.fit(X_scaled, y)
    
    # 2. Random Forest Grid
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 4]
        },
        cv=cv,
        scoring='accuracy'
    )
    rf_grid.fit(X_scaled, y)
    
    # 3. KNN Grid
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid={
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        cv=cv,
        scoring='accuracy'
    )
    knn_grid.fit(X_scaled, y)
    
    tuned_models = {
        "Logistic Regression": {
            "best_estimator": lr_grid.best_estimator_,
            "best_params": lr_grid.best_params_,
            "tuned_cv_acc": lr_grid.best_score_
        },
        "Random Forest": {
            "best_estimator": rf_grid.best_estimator_,
            "best_params": rf_grid.best_params_,
            "tuned_cv_acc": rf_grid.best_score_
        },
        "K-Nearest Neighbors": {
            "best_estimator": knn_grid.best_estimator_,
            "best_params": knn_grid.best_params_,
            "tuned_cv_acc": knn_grid.best_score_
        }
    }
    
    print("\n>>> Tuned 5-Fold Cross-Validation Scores (After Tuning):")
    for name, info in tuned_models.items():
        base_acc = baseline_cv_scores[name] * 100
        tuned_acc = info["tuned_cv_acc"] * 100
        diff = tuned_acc - base_acc
        print(f"  {name:<22}: {tuned_acc:.2f}% (Before: {base_acc:.2f}% | Delta: {diff:+.2f}%)")
        print(f"    Optimal Parameters: {info['best_params']}")
        
    return baseline_cv_scores, tuned_models

def train_test_evaluation(X, y, tuned_models):
    """
    Performs stratified 80/20 train/test evaluation on the tuned models to get
    test set Accuracy, Precision, Recall, F1, and Confusion Matrix.
    """
    print("\n" + "=" * 78)
    print("STEP 3: HELD-OUT TEST SET EVALUATION (80% Train, 20% Test)")
    print("=" * 78)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    test_eval = {}
    
    for name, info in tuned_models.items():
        model = info["best_estimator"]
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        test_eval[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "best_params": info["best_params"],
            "cv_accuracy": info["tuned_cv_acc"]
        }
        
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name} (Tuned):")
        print(f"  Test Accuracy  : {acc*100:.2f}%")
        print(f"  Test Precision : {prec*100:.2f}%")
        print(f"  Test Recall    : {rec*100:.2f}%")
        print(f"  Test F1-Score  : {f1*100:.2f}%")
        print(f"  Confusion Matrix: [TN={tn} FP={fp} | FN={fn} TP={tp}]")
        
    return test_eval, scaler

def serialize_artifacts(test_eval, baseline_cv_scores, feature_names, scaler):
    """
    Selects the best overall model (prioritizing tuned CV and test accuracy),
    and serializes model.pkl, scaler.pkl, and model_meta.json.
    """
    print("\n" + "=" * 78)
    print("STEP 4: MODEL COMPARISON & ARTIFACT SERIALIZATION")
    print("=" * 78)
    
    print(f"{'Model Name':<22} | {'Base CV':<9} | {'Tuned CV':<9} | {'Test Acc':<9} | {'Precision':<9} | {'Recall':<8}")
    print("-" * 78)
    for name, res in test_eval.items():
        base_cv = baseline_cv_scores[name] * 100
        tuned_cv = res["cv_accuracy"] * 100
        test_acc = res["accuracy"] * 100
        prec = res["precision"] * 100
        rec = res["recall"] * 100
        print(f"{name:<22} | {base_cv:>7.2f}% | {tuned_cv:>7.2f}% | {test_acc:>7.2f}% | {prec:>7.2f}% | {rec:>6.2f}%")
    print("-" * 78)
    
    # Pick the model with highest 5-fold CV accuracy (most robust across all 297 records)
    best_name = max(test_eval, key=lambda k: test_eval[k]["cv_accuracy"])
    best_info = test_eval[best_name]
    best_model = best_info["model"]
    
    print(f"\nWINNING MODEL: >>> {best_name} <<<")
    print(f"  5-Fold CV Accuracy : {best_info['cv_accuracy']*100:.2f}%")
    print(f"  Test Set Accuracy  : {best_info['accuracy']*100:.2f}%")
    print(f"  Test Precision     : {best_info['precision']*100:.2f}%")
    print(f"  Test Recall        : {best_info['recall']*100:.2f}%")
    print(f"  Optimal Parameters : {best_info['best_params']}")
    
    # Save model.pkl and scaler.pkl
    joblib.dump(best_model, "model.pkl")
    print("Saved best model to 'model.pkl'.")
    
    joblib.dump(scaler, "scaler.pkl")
    print("Saved StandardScaler to 'scaler.pkl'.")
    
    # Save rich metadata
    metadata = {
        "best_model_name": best_name,
        "best_params": {k: str(v) for k, v in best_info["best_params"].items()},
        "cv_accuracy": round(best_info["cv_accuracy"] * 100, 2),
        "test_accuracy": round(best_info["accuracy"] * 100, 2),
        "precision": round(best_info["precision"] * 100, 2),
        "recall": round(best_info["recall"] * 100, 2),
        "f1_score": round(best_info["f1"] * 100, 2),
        "confusion_matrix": best_info["confusion_matrix"].tolist(),
        "baseline_cv_scores": {k: round(v * 100, 2) for k, v in baseline_cv_scores.items()},
        "comparison_table": {
            name: {
                "baseline_cv": round(baseline_cv_scores[name] * 100, 2),
                "tuned_cv": round(res["cv_accuracy"] * 100, 2),
                "test_accuracy": round(res["accuracy"] * 100, 2),
                "precision": round(res["precision"] * 100, 2),
                "recall": round(res["recall"] * 100, 2),
                "f1_score": round(res["f1"] * 100, 2),
                "best_params": {k: str(v) for k, v in res["best_params"].items()}
            }
            for name, res in test_eval.items()
        },
        "feature_names": feature_names
    }
    
    with open("model_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print("Saved performance metadata to 'model_meta.json'.")
    print("=" * 78)
    print("CROSS-VALIDATION & TUNING PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 78)

def main():
    df = load_data("heart_disease.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    baseline_cv_scores, tuned_models = run_5fold_cross_validation_and_tuning(X, y)
    test_eval, scaler = train_test_evaluation(X, y, tuned_models)
    serialize_artifacts(test_eval, baseline_cv_scores, list(X.columns), scaler)

if __name__ == "__main__":
    main()
