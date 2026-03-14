"""
classify_phase1.py

Compare different classifiers on the raw 132 MediaPipe features.
Tests Random Forest, SVM, KNN, and MLP at 6/20/82 class levels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")


def prep_split(train_features, train_labels, test_features, test_labels, label_col):
    y_train = train_labels[label_col].values # numpy array 
    y_test = test_labels[label_col].values

    scaler = StandardScaler() # scale features to have mean=0 and std=1 for better performance of SVM and MLP
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)
    return X_train, X_test, y_train, y_test

# Random Forest model (bagging)
# 200 trees are trained on a random sample of the training data (rows) drawn with replacement (bootstrap)
# at each split, only a random subset of features (columns) is considered, forcing diversity between trees
# majority vote of the trees is taken as the final prediction (aggregate)
def run_random_forest(X_train, X_test, y_train, y_test):
    # 200 trees and max depth of 30
    model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
    t0 = time()
    model.fit(X_train, y_train)
    train_time = time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Random Forest   Acc: {acc:.4f}  F1: {f1:.4f}  ({train_time:.1f}s)")
    return y_pred, acc, f1, train_time

# SVM model: Find the best boundary (hyperplane) that separates classes with the maximum margin between them
# data points closest to the boundary are called support vectors and determine the position of the boundary
# For nonlinear data, the RBF kernel maps data to a higher-dimensional space to find a linear boundary there
# smoothness of the boundary is controlled by C (higher C = less smooth, more fit to training data) 
# one SVM is trained per class pair 
def run_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)

    t0 = time()
    model.fit(X_train, y_train)
    train_time = time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"SVM    Acc: {acc:.4f}  F1: {f1:.4f}  ({train_time:.1f}s)")
    return y_pred, acc, f1, train_time

# KNN model
# finds 7 nearest neighbors in the training data based on feature distance and predicts the majority class among those neighbors
def run_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)

    t0 = time()
    model.fit(X_train, y_train)
    train_time = time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"KNN   Acc: {acc:.4f}  F1: {f1:.4f}  ({train_time:.1f}s)")
    return y_pred, acc, f1, train_time

# MLP model: NN with 3 hidden layers (256, 128, 64 neurons) and ReLU activation
def run_mlp(X_train, X_test, y_train, y_test):
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )

    t0 = time()
    model.fit(X_train, y_train)
    train_time = time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"MLP    Acc: {acc:.4f}  F1: {f1:.4f}  ({train_time:.1f}s)")
    return y_pred, acc, f1, train_time


def save_confusion_matrix(y_test, y_pred, label_names, title, path):
    # compute confusion matrix and normalize by row (true class) to get percentages
    # each row is the true class, each column is the predicted class
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    n = len(label_names)
    fig_size = max(6, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    show_numbers = n <= 20
    sns.heatmap(
        cm_norm, annot=show_numbers, fmt=".2f" if show_numbers else "",
        xticklabels=label_names, yticklabels=label_names,
        cmap="Blues", ax=ax, vmin=0, vmax=1
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=max(5, 10 - n // 10))
    plt.yticks(rotation=0, fontsize=max(5, 10 - n // 10))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison_chart(results_df, output_dir):
    # 2 plots for accuracy and f1 scores 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # loop through the metrics
    for i, metric in enumerate(["accuracy", "f1"]):
        ax = axes[i]
        # models are rows and label levels are columns, values are the metric (accuracy or f1)
        pivot = results_df.pivot(index="model", columns="level", values=metric)
        col_order = [c for c in ["6-class", "20-class", "82-class"] if c in pivot.columns]
        pivot = pivot[col_order]
        pivot.plot(kind="bar", ax=ax, width=0.75)

        ax.set_title(f"{metric} by Model")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.legend(title="Level")
        ax.tick_params(axis="x", rotation=15)
        for container in ax.containers: 
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2) # add bar labels with the metric value 

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "Phase_1_comparison_chart.png"), dpi=150)
    plt.close(fig)


def main():
    output_dir = "./results/phase1"
    os.makedirs(output_dir, exist_ok=True)

    train_features = np.load("./train_outputs/features.npy")
    train_filtered_data = pd.read_csv("./train_outputs/filtered_rows.csv")
    test_features = np.load("./test_outputs/features.npy")
    test_filtered_data = pd.read_csv("./test_outputs/filtered_rows.csv")

    # which label columns to use for each level
    levels = {
        "6-class": "label_6",
        "20-class": "label_20",
        "82-class": "label_82",
    }

    models = [
        ("Random Forest", run_random_forest),
        ("SVM", run_svm),
        ("KNN", run_knn),
        ("MLP", run_mlp),
    ]

    all_results = []
    for level_name, col in levels.items(): # loop through 6/20/82 class levels
        n_classes = train_filtered_data[col].nunique()
        print(level_name, "-", n_classes, "classes")
        X_train, X_test, y_train, y_test = prep_split(train_features, train_filtered_data, test_features, test_filtered_data, col)
        label_names = sorted(train_filtered_data[col].unique())
        for model_name, func in models:
            y_pred, acc, f1, t = func(X_train, X_test, y_train, y_test)

            # save confusion matrix
            cm_title = f"{model_name} — {level_name} (Acc: {acc:.3f})"
            cm_file = f"cm_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{level_name.replace('-', '_')}.png"
            save_confusion_matrix(y_test, y_pred, label_names, cm_title, os.path.join(output_dir, cm_file))

            all_results.append({
                "model": model_name,
                "level": level_name,
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "train_time": round(t, 2),
            })

    # save results
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string())

    results_df.to_csv(os.path.join(output_dir, "phase_1_results_summary.csv"), index=False)
    save_comparison_chart(results_df, output_dir)


if __name__ == "__main__":
    main()
