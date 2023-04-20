import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

import pandas as pd
from PIL import Image

# reading v 7.3 mat file in python
# https://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python


def convert_jpeg(filepath):
    f = h5py.File(filepath, 'r')  # Open mat file for reading
    cjdata = f['cjdata']  # <HDF5 group "/cjdata" (5 members)>

    # get image member and convert numpy ndarray of type float
    image = np.array(cjdata.get('image')).astype(np.float64)  # In MATLAB: image = cjdata.image

    label = cjdata.get('label')[0, 0]  # Use [0,0] indexing in order to convert lable to scalar

    f.close()

    # Convert image to uint8 (before saving as jpeg - jpeg doesn't support int16 format).
    hi = np.max(image)
    lo = np.min(image)
    image = (((image - lo) / (hi - lo)) * 255).astype(np.uint8)

    # Save as jpeg
    # https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
    # im = Image.fromarray(image)
    # im.save("2.jpg")

    # Display image for testing
    # imgplot = plt.imshow(image)
    # plt.show()

    name = filepath.split('/')[-1].split('.')
    destination = os.path.join(str(int(label)), f'{name[0]}.jpg')
    plt.imsave(destination, image)


def convert_jpg_cv(directory_path):
    directory = os.fsencode(directory_path)

    jpgnames = []
    labels = []
    pids = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filepath = os.path.join(directory_path, filename)
        file_id = filepath.split('/')[-1].split('.')[0]
        jpg_name = file_id + '.jpg'

        f = h5py.File(filepath, 'r')

        cjdata = f['cjdata']

        image = np.array(cjdata.get('image')).astype(np.float64)  # In MATLAB: image = cjdata.image
        label = cjdata.get('label')[0, 0]  # Use [0,0] indexing in order to convert lable to scalar
        PID = cjdata.get('PID')  # <HDF5 dataset "PID": shape (6, 1), type "<u2">
        PID = ''.join(chr(int(c)) for c in PID)

        jpgnames.append(jpg_name)
        labels.append(label)
        pids.append(PID)

        hi = np.max(image)
        lo = np.min(image)
        image = (((image - lo) / (hi - lo)) * 255).astype(np.uint8)

        destination = os.path.join('images_jpg', jpg_name)
        plt.imsave(destination, image)

        f.close()

    labels = pd.DataFrame({'filename': jpgnames, 'label': labels, 'PID': pids})
    print(labels.shape)
    print(labels.head())

    labels.to_csv('labels.csv', index=False)
