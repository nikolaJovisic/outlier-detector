import datetime
from preprocess import preprocess_scan
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from numpy.linalg import norm
import einops
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

path = "data/IORS/png"

model = tf.keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", input_shape=(256, 256, 3), pooling="avg"
)


def clusterize(folder_path, n=None, c="black"):
    i = n if n is not None else float("inf")
    vectors = []
    file_labels = []
    for entity in os.scandir(folder_path):
        if i == 0:
            break
        i -= 1
        file_labels.append(entity.name)
        path = os.path.realpath(entity.path)
        image = cv2.imread(path)[..., 0]
        image, _ = preprocess_scan(image)
        image = einops.repeat(image, "h w -> h w c", c=3)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = cv2.resize(image, (256, 256))
        image = einops.repeat(image, "h w c -> b h w c", b=1)
        feature_vector = model.predict(image).flatten()
        vectors.append(feature_vector)

    features = np.array(vectors)
    pca = PCA(n_components=16)
    X = pca.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    distances = cdist(X, centers, "euclidean")
    min_distances = np.min(distances, axis=1)
    threshold = 15.0
    outliers = min_distances > threshold

    print(min_distances)

    class_dict = {}
    outlier_list = []

    for string, class_, is_outlier in zip(file_labels, labels, outliers):
        if is_outlier:
            outlier_list.append(string)
        else:
            if class_ not in class_dict:
                class_dict[class_] = []
            class_dict[class_].append(string)
    return class_dict, outlier_list


def write_report(class_dict, outlier_list):
    with open(
        f"cluster_logs/log_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt",
        "w",
    ) as file:
        # Write outliers
        file.write("outliers\n")
        for outlier in outlier_list:
            file.write(f"{outlier}\n")
        file.write("\n")

        # Write each class
        for class_, items in class_dict.items():
            file.write(f"class_{class_}\n")
            for item in items:
                file.write(f"{item}\n")
            file.write("\n")


write_report(*clusterize(path))
