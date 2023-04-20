import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.feature_extraction import create_feature_extractor


# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label


def extract(data_path, model_path):
    BATCH_SIZE = 30

    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.Lambda(lambda image: image.convert('RGB')),
        T.ToTensor(),
        T.Lambda(lambda image: (image - image.min()) / (image.max() - image.min()) * (1 - 0) + 0)
    ])

    imgfolder = torchvision.datasets.ImageFolder(root=data_path, transform=img_transform)

    dataloader = torch.utils.data.DataLoader(imgfolder, batch_size=BATCH_SIZE, shuffle=False)
    # total_count = len(imgfolder)

    # dataset = CustomImageDataset('labels.csv', 'images_jpg')
    # train = Subset(dataset, train_index)
    # test = Subset(dataset, test_index)
    # dataloader = torch.utils.data.DataLoader(train, batch_size=16)

    # Make transformation call here if using data augmentation
    # train_dataset.dataset.transform = img_transform

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.googlenet(weights='IMAGENET1K_V1').to(device)
    model.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True).to(device)
    model.load_state_dict(torch.load(model_path))

    return_nodes = {'avgpool': 'pooling' }
    extractor = create_feature_extractor(model, return_nodes=return_nodes).to(device)

    deep_cnn_features = []
    y_true = []
    f = open('img_paths.txt', 'w')
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader, 0):
            imgs = imgs.to(device)
            labels = labels.to(device)

            intermediate_outputs = extractor(imgs)
            deep_cnn_features.append(intermediate_outputs['pooling'].squeeze().to('cpu'))

            y_true.extend(labels.tolist())
            sample_fname, _ = dataloader.dataset.samples[i]
            f.write(f'{sample_fname}\n')

    f.close()

    deep_cnn_features = np.hstack((np.vstack(deep_cnn_features), np.array(y_true).reshape(-1,1)))

    print(deep_cnn_features.shape)
    np.savetxt('deep_cnn_features.csv', deep_cnn_features, delimiter=',')

    return
