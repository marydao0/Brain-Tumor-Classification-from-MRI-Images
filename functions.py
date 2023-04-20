import pickle
import time
import torch
import torchvision
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import torchvision.transforms as T
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import pandas as pd
import numpy as np

def train_epoch(model, device, train_dataloader, loss_fn, optimizer):
    model.train()

    losses = 0
    train_item_counter = 0
    for imgs, labels in train_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        with open('train_losses.txt', 'a') as f:
            f.write(f'{loss}\n')

        losses += loss.item()
        train_item_counter += 1

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


def run_inferences(model, device, test_dataloader, save_y):
    model.eval()

    y_true = []
    y_pred = []
    for imgs, labels in test_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        y_true.extend(labels.tolist())

        outputs = model(imgs)
        y_pred.extend(torch.argmax(outputs, dim=1).tolist())

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'F1 score: {f1}')

    if save_y:
        with open('y_true', 'wb') as fp:
            pickle.dump(y_true, fp)

        with open('y_pred', 'wb') as fp:
            pickle.dump(y_pred, fp)

    return


def train_googlenet(path):
    # Set hyperparameters
    BATCH_SIZE = 30
    NUM_EPOCHS = 10
    LR = 0.0003

    img_transform = T.Compose([
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        T.Resize((224, 224)),
        T.Lambda(lambda image: image.convert('RGB')),
        T.ToTensor(),
        T.Lambda(lambda image: (image - image.min()) / (image.max() - image.min()) * (1 - 0) + 0)
    ])

    imgfolder = torchvision.datasets.ImageFolder(root=path, transform=img_transform)
    total_count = len(imgfolder)

    # Split into 70/10/20 train/val/test
    train_count = int(0.7 * total_count)
    valid_count = int(0.1 * total_count)
    test_count = total_count - train_count - valid_count

    gen = torch.Generator().manual_seed(3)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        imgfolder, (train_count, valid_count, test_count),
        generator=gen
    )

    # Make transformation call here if using data augmentation
    # train_dataset.dataset.transform = img_transform

    # Define dataloaders
    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_dataset_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    dataloaders = {
        'train': train_dataset_loader,
        'val': valid_dataset_loader,
        'test': test_dataset_loader,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.googlenet(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    min_val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, device, dataloaders['train'], loss_fn, optimizer)
        end_time = time.time()

        val_loss = evaluate(model, device, dataloaders['val'], loss_fn)

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")

        with open('train_results.txt', 'a') as file:
            file.write(f'{epoch},{train_loss},{val_loss},{(end_time - start_time)}\n')

        if val_loss < min_val_loss:
            torch.save(model.state_dict(), 'googlenet_best.pt')
            min_val_loss = val_loss

    model.load_state_dict(torch.load('googlenet_best.pt'))
    run_inferences(model, device, dataloaders['test'], save_y=True)


def dim_red(train_path, test_path):
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)

    X_train = train.iloc[:, :-1].to_numpy()
    y_train = train.iloc[:, -1].to_numpy()
    X_test = test.iloc[:, :-1].to_numpy()
    y_test = test.iloc[:, -1].to_numpy()

    pca = PCA(n_components=2)
    X_red = pd.DataFrame(pca.fit_transform(X_train), columns=['Principal Component 1', 'Principal Component 2'])
    subtypes = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary'}
    X_red['Subtype'] = [subtypes[i] for i in y_train]

    fig, ax = plt.subplots()
    sns.scatterplot(data=X_red, x='Principal Component 1', y='Principal Component 2',
                    hue='Subtype', ax=ax, palette='Set2')

    plt.show()
    return


def plot_results(train_path, val_path):
    train = pd.read_csv(train_path, header=None)
    val = pd.read_csv(val_path, header=None)

    n = len(val)
    epochs = list(range(0, n, n // 10))

    fig, ax = plt.subplots()
    # ax.plot(train[0], color='midnightblue')
    ax.plot(val)
    [ax.axvline(x=i, color='black', linestyle='dashed') for i in epochs]
    ax.set_xlim(0, n)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    # ax.plot(df[0], df[2], label='validation', linestyle='dotted')
    plt.show()


def plot_features(path):
    df = pd.read_csv(path, header=None)

    df = pd.read_csv('deep_cnn_features.csv', header=None)
    # labels = {0: 'meningioma', 1: 'glioma', 2: 'pituitary'}
    # l = [labels[i] for i in df[1024]]

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    sc = ax.scatter(df[0], df[1], df[3], c=df[1024])

    xLabel = ax.set_xlabel('Feature 1')
    yLabel = ax.set_ylabel('Feature 2')
    zLabel = ax.set_zlabel('Feature 3')
    ax.legend(*sc.legend_elements())

    plt.show()
    return

def plot_gridsearch(gs_svm, gs_knn):
    # gs_knn = [0.86979592, 0.94530612, 0.94612245, 0.94612245, 0.94530612, 0.94612245, 0.9444898, 0.94489796, 0.94571429, 0.94530612, 0.94530612]
    # gs = [0.80367347, 0.93265306, 0.92693878, 0.92612245, 0.94857143, 0.9244898, 0.92897959, 0.92734694, 0.90734694, 0.88530612, 0.93142857]
    # plt.scatter(N_FEATURES_OPTIONS,gs)
    fig, ax = plt.subplots()
    ax.plot(gs_svm, marker='o', color='mediumaquamarine', label='SVM')
    ax.plot(gs_knn, marker='o', color='slateblue', label='KNN')
    # ax.annotate('', xy=(4, 0.945), xytext=(4,0.93),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_xticks(ticks=range(len(gs)),labels=N_FEATURES_OPTIONS)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.tight_layout()
    plt.show()


def gridsearch_pca(clf, X, y):
    pipe = Pipeline(
        [
            ("reduce_dim", "passthrough"),
            ("classify", clf)
        ]
    )

    N_FEATURES_OPTIONS = [2 ** i for i in range(11)]

    param_grid = [
        {
            "reduce_dim": [PCA()],
            "reduce_dim__n_components": N_FEATURES_OPTIONS
        }
    ]

    grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
    grid.fit(X, y)

    print(np.array(grid.cv_results_["mean_test_score"]))
    print(grid.best_params_)


def train_googlenet_cv(path):
    # Set hyperparameters
    BATCH_SIZE = 30
    NUM_EPOCHS = 10
    LR = 0.0003

    img_transform = T.Compose([
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        T.Resize((224, 224)),
        T.Lambda(lambda image: image.convert('RGB')),
        T.ToTensor(),
        T.Lambda(lambda image: (image - image.min()) / (image.max() - image.min()) * (1 - 0) + 0)
    ])

    imgfolder = torchvision.datasets.ImageFolder(root=path, transform=img_transform)
    kf = KFold(n_splits=5, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_acc = []
    min_loss = float('inf')
    for i, (train_index, test_index) in enumerate(kf.split(imgfolder)):
        train = torch.utils.data.Subset(imgfolder, train_index)
        test = torch.utils.data.Subset(imgfolder, test_index)

        train_dataset_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

        dataloaders = {
            'train': train_dataset_loader,
            'test': test_dataset_loader,
        }

        model = torchvision.models.googlenet(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(1, NUM_EPOCHS + 1):
            start_time = time.time()
            train_loss = train_epoch(model, device, dataloaders['train'], loss_fn, optimizer)
            end_time = time.time()

            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")

            with open('train_results.txt', 'a') as file:
                file.write(f'{epoch},{train_loss},{(end_time - start_time)}\n')

            if train_loss < min_loss:
                torch.save(model.state_dict(), 'googlenet_best_cv.pt')
                min_loss = train_loss

        model.load_state_dict(torch.load('googlenet_best_cv.pt'))
        run_inferences(model, device, dataloaders['test'], save_y=False)

def classify_cv(paths, clf):
    train_path = paths['train'] #'deep_cnn_features.csv'
    test_path = paths['test'] #'test_deep_cnn_features.csv'
    val_path = paths['val'] #'val_deep_cnn_features.csv'

    train = pd.read_csv(train_path, header=None)
    val = pd.read_csv(val_path, header=None)
    test = pd.read_csv(test_path, header=None)

    X_train = train.iloc[:, :-1].to_numpy()
    X_val = val.iloc[:, :-1].to_numpy()
    X_test = test.iloc[:, :-1].to_numpy()
    # X_train = PCA(n_components=16).fit_transform(train.iloc[:, :-1].to_numpy())
    # X_val = PCA(n_components=16).fit_transform(val.iloc[:, :-1].to_numpy())
    # X_test = PCA(n_components=16).fit_transform(test.iloc[:, :-1].to_numpy())

    y_train = train.iloc[:, -1].to_numpy().reshape(-1,1)
    y_val = val.iloc[:, -1].to_numpy().reshape(-1,1)
    y_test = test.iloc[:, -1].to_numpy().reshape(-1,1)

    X = np.vstack((X_train, X_val, X_test))
    y = np.vstack((y_train, y_val, y_test)).reshape(-1,)

    # knn = KNeighborsClassifier(n_neighbors=49, weights='distance')
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)

    total_acc = []
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
        ### Dividing data into folds
        x_train_fold = X[train_index]
        x_test_fold = X[test_index]
        y_train_fold = y[train_index]
        y_test_fold = y[test_index]

        x_train_fold = PCA(n_components=16).fit_transform(x_train_fold)
        x_test_fold = PCA(n_components=16).fit_transform(x_test_fold)

        # knn = KNeighborsClassifier(n_neighbors=49, weights='distance')
        # knn.fit(x_train_fold, y_train_fold)
        # y_pred = knn.predict(x_test_fold)
        # clf = make_pipeline(StandardScaler(),
        #                     svm.LinearSVC(random_state=0, loss='hinge', max_iter=100000))
        clf.fit(x_train_fold, y_train_fold)
        y_pred = clf.predict(x_test_fold)

        total_acc.append(sklearn.metrics.accuracy_score(y_test_fold, y_pred))
    avg_acc = (sum(total_acc) / kfold.get_n_splits())
    print(np.mean(total_acc))
    std = np.std(total_acc)

    print(f'Average 5-Fold Cross Validation Accuracy: {avg_acc}')
    print(f'Standard Deviation: {std}')


def cv_fit():
    cv_splits = []
    for fold in folds:
        train_index = img_paths.loc[~img_paths.PID.isin(fold), 'index'].to_numpy()
        test_index = img_paths.loc[img_paths.PID.isin(fold), 'index'].to_numpy()

        cv_splits.append((train_index, test_index))

    max = -float('inf')
    scores = []
    for i in range(1,50):
        cv_score = cross_val_score(KNeighborsClassifier(n_neighbors=i, weights='distance'), X, y, cv=cv_splits)
        cv_avg = np.mean(cv_score)
        scores.append(cv_avg)
        if cv_avg > max:
            print(i)
            print(cv_score)
            print(cv_avg)
            max = cv_avg


def pca_classify():
    def classify_pca(clf, X_train, y_train, X_test):
        clf.fit(X_train, y_train)
        return clf.predict(X_test)

    fig, axs = plt.subplots(2, 5, sharey='row', sharex='col')

    knn_acc = []
    svm_acc = []
    for fold, train_ax, test_ax in zip(list(range(5)), axs[0], axs[1]):
        train = pd.read_csv(f'results_pt_cv/fold{fold}_train_deep_cnn_features.csv',header=None)
        test = pd.read_csv(f'results_pt_cv/fold{fold}_test_deep_cnn_features.csv', header=None)

        X_train = train.iloc[:, :-1]
        y_train = train.iloc[:, -1]

        X_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]

        pca = PCA(n_components=2)

        X_train_red = pca.fit_transform(X_train)
        X_test_red = pca.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=49)
        svm = SVC(kernel='linear')

        knn_pred = classify_pca(knn, X_train_red, y_train, X_test_red)
        svm_pred = classify_pca(svm, X_train_red, y_train, X_test_red)

        knn_acc.append(accuracy_score(y_test, knn_pred))
        svm_acc.append(accuracy_score(y_test, svm_pred))

        train_df = pd.DataFrame({'PC1': X_train_red[:, 0], 'PC2': X_train_red[:, 1], 'label': y_train})
        sns.scatterplot(data=train_df, x='PC1', y='PC2', hue='label', ax=train_ax, legend=False)
        train_ax.set_title(f'Fold {fold+1}')

        test_df = pd.DataFrame({'PC1': X_test_red[:, 0], 'PC2': X_test_red[:, 1], 'label': y_test})
        ax = sns.scatterplot(data=test_df, x='PC1', y='PC2', hue='label', ax=test_ax)
        ax.get_legend().remove()

    handles, labels = ax.get_legend_handles_labels()
    labels = ['Meningioma', 'Glioma', 'Pituitary']
    fig.legend(handles, labels, loc='upper right', ncol=3, bbox_to_anchor=(0.72, 0.98))
    axs[0][0].set_title(f'Training Fold 1')
    axs[1][0].set_title('Testing')

    print(knn_acc)
    print(svm_acc)

    fig2, ax2 = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, knn_pred, display_labels=labels, ax=ax2)
    fig2.tight_layout()
    plt.show()
