import numpy as np
import pandas as pd
import os
import pickle
import torch
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import torchvision
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.models.feature_extraction import create_feature_extractor


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train_epoch(model, device, train_dataloader, loss_fn, optimizer, fold):
    model.train()

    losses = 0
    train_item_counter = 0
    f = open(f'fold{fold}_train_losses.txt', 'w')
    for imgs, labels in train_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        f.write(f'{loss}\n')

        losses += loss.item()
        train_item_counter += 1
    f.close()

    return losses / train_item_counter


def evaluate(model, device, val_dataloader, loss_fn):
    model.eval()

    losses = 0
    val_item_counter = 0
    for imgs, labels in val_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)

        loss = loss_fn(outputs, labels)

        with open('val_losses.txt', 'a') as f:
            f.write(f'{loss}\n')

        losses += loss.item()
        val_item_counter += 1

    return losses / val_item_counter


def run_inferences(model, device, test_dataloader, save_y, fold):
    model.eval()

    y_true = []
    y_pred = []
    for imgs, labels in test_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        y_true.extend(labels.tolist())

        outputs = model(imgs)
        y_pred.extend(torch.argmax(outputs, dim=1).tolist())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Fold {fold} GoogLeNet Accuracy: {accuracy}')
    print(f'Fold {fold} GoogLeNet F1 score: {f1}\n')

    if save_y:
        with open(f'fold{fold}_cnn_y_true', 'wb') as fp:
            pickle.dump(y_true, fp)

        with open(f'fold{fold}_cnn_y_pred', 'wb') as fp:
            pickle.dump(y_pred, fp)

    return accuracy


def extract_features(model, device, dataloader, datasplit, fold):
    model.eval()
    return_nodes = {'avgpool': 'pooling'}
    extractor = create_feature_extractor(model, return_nodes=return_nodes).to(device)

    deep_cnn_features = []
    y_true = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            intermediate_outputs = extractor(imgs)
            deep_cnn_features.append(intermediate_outputs['pooling'].squeeze().to('cpu'))

            y_true.extend(labels.tolist())

    deep_cnn_features = np.hstack((np.vstack(deep_cnn_features), np.array(y_true).reshape(-1, 1)))
    np.savetxt(f'fold{fold}_{datasplit}_deep_cnn_features.csv', deep_cnn_features, delimiter=',')

    return deep_cnn_features


def classify_features(clf, clf_name, train_features, test_features, save_y, fold):
    X_train = train_features[:, :-1]
    X_test = test_features[:, :-1]

    y_train = train_features[:, -1]
    y_test = test_features[:, -1]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Fold {fold} {clf_name} Accuracy: {accuracy}')
    print(f'Fold {fold} {clf_name} F1 score: {f1}\n')

    if save_y:
        with open(f'fold{fold}_{clf_name}_y_true', 'wb') as fp:
            pickle.dump(y_test, fp)

        with open(f'fold{fold}_{clf_name}_y_pred', 'wb') as fp:
            pickle.dump(y_pred, fp)

    return accuracy


def classify_cv(annotations, img_path):
    # Set hyperparameters
    BATCH_SIZE = 30
    NUM_EPOCHS = 10
    LR = 0.0003
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_transform = T.Compose([
        T.Resize((224, 224), antialias=False),
        T.Lambda(lambda image: (image - image.min()) / (image.max() - image.min()) * (1 - 0) + 0)
    ])

    dataset = CustomImageDataset(annotations_file=annotations, img_dir=img_path, transform=img_transform)

    # Getting indices for patient_level cross-validation
    img_metadata = dataset.img_labels.copy().reset_index(drop=False)
    pid_unique = img_metadata.PID.unique().tolist()
    n_pids = len(pid_unique)
    indices = list(range(0, n_pids, n_pids//5))
    indices[-1] = n_pids+1
    folds = [pid_unique[indices[i]:j] for i, j in enumerate(indices[1:])]

    googlenet_all_acc = []
    svm_all_acc = []
    knn_all_acc = []
    for i, fold in enumerate(folds):
        train_index = img_metadata.loc[~img_metadata.PID.isin(fold), 'index'].to_numpy()
        test_index = img_metadata.loc[img_metadata.PID.isin(fold), 'index'].to_numpy()

        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = torchvision.models.googlenet(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        min_loss = float('inf')
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_epoch(model, device, train_loader, loss_fn, optimizer, i)

            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}")

            with open('train_results.txt', 'a') as file:
                file.write(f'{epoch},{train_loss}\n')

            if train_loss < min_loss:
                torch.save(model.state_dict(), f'fold{i}_googlenet_best.pt')
                min_loss = train_loss

        model.load_state_dict(torch.load('googlenet_best.pt'))
        googlenet_acc = run_inferences(model, device, test_loader, True, i)
        googlenet_all_acc.append(googlenet_acc)

        train_features = extract_features(model, device, train_loader, 'train', i)
        test_features = extract_features(model, device, test_loader, 'test', i)

        svm = SVC(kernel='linear')
        svm_acc = classify_features(svm, 'SVM', train_features, test_features, True, i)
        svm_all_acc.append(svm_acc)

        knn = KNeighborsClassifier(n_neighbors=49)
        knn_acc = classify_features(knn, 'KNN', train_features, test_features, True, i)
        knn_all_acc.append(knn_acc)

    print('GoogLeNet Average:', np.mean(googlenet_all_acc))
    print('GoogLeNet Average:', np.std(googlenet_all_acc), '\n')

    print('SVM Average:', np.mean(svm_all_acc))
    print('SVM Average:', np.std(svm_all_acc), '\n')

    print('KNN Average:', np.mean(knn_all_acc))
    print('KNN Average:', np.std(knn_all_acc))
