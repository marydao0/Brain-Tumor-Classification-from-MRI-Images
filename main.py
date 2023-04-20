import pipeline
import torch
import pandas as pd
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset, DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import functions as f

if __name__ == '__main__':
    # df = pd.read_csv('cv_acc.csv')
    # print(df.head())
    # fig, ax = plt.subplots()
    # sns.barplot(data=df, x='Model', y='Mean Accuracy', hue='CV Split Method', ax=ax, palette='YlGnBu', linewidth=1, edgecolor=".5")
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.tight_layout()
    # plt.show()
    f.pca_classify()
