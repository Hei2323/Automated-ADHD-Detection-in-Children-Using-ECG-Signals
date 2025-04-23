import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


class ECGDataset(Dataset):
    def __init__(self, ecg_data, labels, age, gender, features):
        self.ecg_data = ecg_data
        self.labels = labels
        self.age = age
        self.gender = gender
        self.features = features

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg = self.ecg_data[idx]
        label = self.labels[idx]
        age = self.age[idx]
        gender = self.gender[idx]
        feature = self.features[idx]
        return ecg, label, age, gender, feature


def get_kfold_dataloaders(ecg_data, labels, age, gender, features, batch_size=64, n_splits=10):
    """
    Returns data loaders for K-fold cross-validation.

    :param ecg_data: ECG data (numpy array or torch tensor)
    :param labels: Labels for the data
    :param age: Age information for the samples
    :param gender: Gender information for the samples
    :param features: Feature array
    :param batch_size: Size of the batches for DataLoader
    :param n_splits: Number of splits for cross-validation (default: 10)
    :return: List of (train_loader, test_loader) tuples for each fold
    """
    dataset = ECGDataset(ecg_data, labels, age, gender, features)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    dataloaders = []

    for train_index, test_index in kf.split(ecg_data, labels):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        dataloaders.append((train_loader, test_loader))

    return dataloaders
