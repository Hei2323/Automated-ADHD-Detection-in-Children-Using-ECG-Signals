import torch
import torch.optim as optim
from models.adhd_1d_cnn import CNN_ADHD
from feature_extraction.score_cam import ScoreCAM
from feature_extraction.feature_extractor import extract_features
from utils.data_loader import get_kfold_dataloaders
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
def load_data():
    ecg_data = torch.tensor(np.load("ecg_data.npy")).float()
    labels = torch.tensor(np.load("labels.npy")).long()
    age = torch.tensor(np.load("age.npy")).float()
    gender = torch.tensor(np.load("gender.npy")).long()
    features = np.load("features.npy")
    return ecg_data, labels, age, gender, features

# Train 1D-CNN
def train_cnn(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0
    for inputs, labels, _, _, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = corrects / total

    return epoch_loss, epoch_acc

# Main training cycle - 1D-CNN training
def train_cnn_main(ecg_data, labels, age, gender, features):
    dataloaders = get_kfold_dataloaders(ecg_data, labels, age, gender, features, batch_size=64, n_splits=10)

    for fold, (train_loader, _) in enumerate(dataloaders):
        print(f"Training fold {fold + 1} with 1D-CNN")

        model = CNN_ADHD().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)

        # Train 1D-CNN
        for epoch in range(1, 101):
            train_loss, train_acc = train_cnn(model, train_loader, criterion, optimizer)
            print(f"Epoch {epoch} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # save model
        torch.save(model.state_dict(), f"cnn_adhd_fold_{fold + 1}.pth")

# Main
def main():
    ecg_data, labels, age, gender, features = load_data()
    print("Training CNN Model...")
    train_cnn_main(ecg_data, labels, age, gender, features)

if __name__ == "__main__":
    main()
