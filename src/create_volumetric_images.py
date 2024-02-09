import cv2
import numpy as np
import os

from glob import glob
from pathlib import Path


def get_x_paths_and_y_paths_1(x_parent_path, y_parent_path):
    def f():
        for y_path in map(Path, sorted(glob(f"{y_parent_path}/*.tif"))):
            x_path = x_parent_path / y_path.name

            yield x_path, y_path

    return zip(*f())


def get_x_paths_and_y_paths_2(x_parent_path, y_parent_path_1, y_parent_path_2):
    def f():
        for x_path in map(Path, sorted(glob(f"{x_parent_path}/*.tif"))):
            y_path = y_parent_path_1 / x_path.name if (y_parent_path_1 / x_path.name).exists() else y_parent_path_2 / x_path.name

            x_path = x_parent_path / y_path.name

            yield x_path, y_path

    return zip(*f())


def create_volumetric_image(paths, scale):
    result = np.stack(tuple(map(lambda path: cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), None, None, scale, scale), paths)))
    result = np.transpose(result, (1, 2, 0))
    result = np.stack(tuple(map(lambda image: cv2.resize(image, None, None, scale, 1), result)))
    result = np.transpose(result, (2, 0, 1))
    result = (result / 255).astype(np.float32)

    return result


os.makedirs('../data/volumetric_images/all', exist_ok=True)

for i, ((x_paths, y_paths), scale) in enumerate(((get_x_paths_and_y_paths_1(Path('../data/blood-vessel-segmentation/train/kidney_1_dense/images'),
                                                                            Path('../data/blood-vessel-segmentation/train/kidney_1_dense/labels')),
                                                  50 / 63.08),
                                                 (get_x_paths_and_y_paths_1(Path('../data/blood-vessel-segmentation/train/kidney_2/images'),
                                                                            Path('../data/blood-vessel-segmentation/train/kidney_2/labels')),
                                                  50 / 63.08),
                                                 (get_x_paths_and_y_paths_2(Path('../data/blood-vessel-segmentation/train/kidney_3_sparse/images'),
                                                                            Path('../data/blood-vessel-segmentation/train/kidney_3_dense/labels'),
                                                                            Path('../data/blood-vessel-segmentation/train/kidney_3_sparse/labels')),
                                                  50.16 / 63.08))):
    x = create_volumetric_image(x_paths, scale)
    y = create_volumetric_image(y_paths, scale)

    y[y >= 0.5] = 1
    y[y <  0.5] = 0  # noqa: E222

    np.savez(f"../data/volumetric_images/all/{i:04}", x=x, y=y)


os.makedirs('../data/volumetric_images/dense', exist_ok=True)

for i, ((x_paths, y_paths), scale) in enumerate(((get_x_paths_and_y_paths_1(Path('../data/blood-vessel-segmentation/train/kidney_1_dense/images'),
                                                                            Path('../data/blood-vessel-segmentation/train/kidney_1_dense/labels')),
                                                  50 / 63.08),
                                                 (get_x_paths_and_y_paths_1(Path('../data/blood-vessel-segmentation/train/kidney_3_sparse/images'),
                                                                            Path('../data/blood-vessel-segmentation/train/kidney_3_dense/labels')),
                                                  50.16 / 63.08))):
    x = create_volumetric_image(x_paths, scale)
    y = create_volumetric_image(y_paths, scale)

    y[y >= 0.5] = 1
    y[y <  0.5] = 0  # noqa: E222

    np.savez(f"../data/volumetric_images/dense/{i:04}", x=x, y=y)
