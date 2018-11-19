from time import time
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA


def plot_gallery(images, titles, eigenvalues, meanface, reconstruct, residual, h, w, n_row=2, n_col=3):
    plt.figure(figsize=(1.8 * n_col, 2.2 * (n_row + 1)))
    plt.suptitle("K = " + str(n_row*n_col), va='bottom', ha='right', x=0.85, y=(1-n_row/(n_row+1))/2.5, size=36)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot((n_row + 1), n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i]+'\n'+str("{0:.2f}".format(eigenvalues[i])), size=12)
        plt.xticks(())
        plt.yticks(())
    plt.subplot((n_row + 1), n_col, n_row*n_col+1)
    plt.imshow(meanface.reshape((h, w)), cmap=plt.cm.gray)
    plt.title('Mean Face', size=12)
    plt.xticks(())
    plt.yticks(())

    plt.subplot((n_row + 1), n_col, n_row * n_col + 2)
    plt.imshow(reconstruct.reshape((h, w)), cmap=plt.cm.gray)
    plt.title('Reconstruct', size=12)
    plt.xticks(())
    plt.yticks(())

    plt.subplot((n_row + 1), n_col, n_row * n_col + 3)
    plt.imshow(residual.reshape((h, w)), cmap=plt.cm.gray)
    plt.title('Residual', size=12)
    plt.xticks(())
    plt.yticks(())


test_img = Image.open('data/hw03-test.tif')

train_imgs = []
train_label = []

# Retrieve All Tif file
for filename in glob.glob('data/training.db/*.tif'):
    img = Image.open(filename)
    label = os.path.splitext(os.path.basename(filename))[0]
    train_imgs.append(np.asarray(img))
    train_label.append(label)

X_arr = np.array(train_imgs)
y_arr = np.array(train_label)

X_test_arr = np.array(test_img)

# Reshape Data
n_samples, h, w = X_arr.shape
X = X_arr.reshape(n_samples, h*w)

X_test = X_test_arr.reshape(1, 128*128)

n_features = X.shape[1]

for n_components in range(5, 26, 5):
    print("K = ", n_components)
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, whiten=True).fit(X)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))

    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]

    # Center the data and compute the sample covariance matrix.
    eigenmean = pca.mean_
    eigenvalues = pca.explained_variance_
    # print(pca.explained_variance_ratio_)

    print("done in %0.3fs" % (time() - t0))
    result = []

    X_test_pca = pca.transform(X_test)

    reconstruct = pca.inverse_transform(X_test_pca)

    residual = X_test - reconstruct

    n_row = int(n_components / 5)
    n_col = 5

    # Mean Face = eigenmean
    plot_gallery(eigenfaces, eigenface_titles, eigenvalues, eigenmean, reconstruct, residual, h, w, n_row=n_row, n_col=n_col)
    plt.show()



