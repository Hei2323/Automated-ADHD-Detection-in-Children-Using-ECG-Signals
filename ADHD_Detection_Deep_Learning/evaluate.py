import torch
from models.adhd_1d_cnn import CNN_ADHD
from feature_extraction.score_cam import ScoreCAM
from feature_extraction.feature_extractor import extract_features
from utils.data_loader import get_kfold_dataloaders
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data():
    ecg_data = torch.tensor(np.load("ecg_data.npy")).float()
    labels = torch.tensor(np.load("labels.npy")).long()
    age = torch.tensor(np.load("age.npy")).float()
    gender = torch.tensor(np.load("gender.npy")).long()
    features = np.load("features.npy")
    return ecg_data, labels, age, gender, features

# Test the CNN model
def test_cnn(model, test_loader):
    model.eval()
    corrects = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels, _, _, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = corrects / total
    return accuracy, y_true, y_pred

# Evaluate the CNN model
def evaluate_cnn(ecg_data, labels, age, gender, features):
    dataloaders = get_kfold_dataloaders(ecg_data, labels, age, gender, features, batch_size=64, n_splits=10)

    for fold, (train_loader, test_loader) in enumerate(dataloaders):
        print(f"Evaluating fold {fold + 1} with CNN")

        model = CNN_ADHD().to(device)
        model.load_state_dict(torch.load(f"cnn_adhd_fold_{fold + 1}.pth"))

        test_acc, y_true, y_pred = test_cnn(model, test_loader)
        print(f"CNN Test Accuracy: {test_acc:.4f}")

        print("CNN Classification Report:")
        print(classification_report(y_true, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title(f"Confusion Matrix - CNN - Fold {fold + 1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"confusion_matrix_cnn_fold_{fold + 1}.png", format="png", dpi=300, bbox_inches="tight")

# Evaluate the CNN + Score-CAM + machine learning model
def evaluate_cnn_ml(ecg_data, labels, age, gender, features):
    dataloaders = get_kfold_dataloaders(ecg_data, labels, age, gender, features, batch_size=64, n_splits=10)

    # Obtain the machine learning classifier
    classifiers = {
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "DecisionTree": DecisionTreeClassifier()
    }

    for fold, (train_loader, test_loader) in enumerate(dataloaders):
        print(f"Evaluating fold {fold + 1} with CNN + Score-CAM + Machine Learning")

        model = CNN_ADHD().to(device)
        model.load_state_dict(torch.load(f"cnn_adhd_fold_{fold + 1}.pth"))

        # Extract Features
        score_cam = ScoreCAM(model, model.conv4)  # Select the target convolutional layer
        train_features = []
        y_train = []
        test_features = []
        y_test = []

        # Extract the features of the train and test data
        model.eval()
        with torch.no_grad():
            for inputs, labels, _, _, _ in train_loader:
                inputs = inputs.to(device)
                target_class = labels
                heatmap = score_cam.forward(inputs, target_class)
                feature_vector = extract_features(heatmap)
                train_features.append(feature_vector)
                y_train.extend(labels.cpu().numpy())

            for inputs, labels, _, _, _ in test_loader:
                inputs = inputs.to(device)
                target_class = labels
                heatmap = score_cam.forward(inputs, target_class)
                feature_vector = extract_features(heatmap)
                test_features.append(feature_vector)
                y_test.extend(labels.cpu().numpy())

        train_features = np.vstack(train_features)
        test_features = np.vstack(test_features)
        # Standardize the characteristics
        scaler = RobustScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.fit_transform(test_features)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Train and evaluate machine learning classifiers
        for clf_name, clf in classifiers.items():
            print(f"Evaluating with {clf_name}")
            clf.fit(train_features_scaled, y_train)
            preds = clf.predict(test_features_scaled)

            # Print the classification report
            print(f"{clf_name} Classification Report:")
            print(classification_report(y_test, preds, digits=4))

            # Calculate the accuracy
            accuracy = accuracy_score(y_test, preds)
            print(f"{clf_name} Accuracy: {accuracy:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            plt.title(f"Confusion Matrix - {clf_name} - Fold {fold + 1}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(f"confusion_matrix_{clf_name}_fold_{fold + 1}.png", format="png", dpi=300, bbox_inches="tight")

# 主评估函数
def evaluate():
    # Load data
    ecg_data, labels, age, gender, features = load_data()
    # Evaluate the CNN model
    evaluate_cnn(ecg_data, labels, age, gender, features)
    # Evaluate the CNN + Score-CAM + machine learning model
    evaluate_cnn_ml(ecg_data, labels, age, gender, features)

if __name__ == "__main__":
    evaluate()
