import datetime
import os
import cv2
import sys
import einops
import numpy as np
from sklearn.decomposition import PCA
import json
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

path = "data"
run_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def embed(folder_path, n=None, c="black"):
    import pydicom
    from preprocess import preprocess_scan
    import tensorflow as tf
    
    model = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(256, 256, 3), pooling="avg"
    )

    i = n if n is not None else float("inf")
    vectors = []
    file_labels = []
    for entity in os.scandir(folder_path):
        if i == 0:
            break
        i -= 1
        file_labels.append(entity.name)
        print(entity.path)
        path = os.path.realpath(entity.path)
        print(path)
        image = pydicom.dcmread(path).pixel_array
        image, _ = preprocess_scan(image)
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        image = einops.repeat(image, "h w -> h w c", c=3)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = cv2.resize(image, (256, 256))
        image = einops.repeat(image, "h w c -> b h w c", b=1)
        feature_vector = model.predict(image).flatten()
        vectors.append(feature_vector)

    features = np.array(vectors)
    os.mkdir(f'runs/{run_id}')
    np.save(f'runs/{run_id}/features.npy', features)
    with open(f'runs/{run_id}/names.json', 'w') as file:
        json.dump(file_labels, file)
    return features, file_labels

def clusterize(features, file_labels):
    pca = PCA(n_components=6)
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
        f"cluster_logs/log_{run_id}.txt",
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

def main(arg=None):
    if arg:
        with open(f'{arg}/names.json', 'r') as file:
            names = json.load(file)
        write_report(*clusterize(np.load(f'{arg}/features.npy'), names))
    else:
        write_report(*clusterize(*embed(path, n=10)))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
