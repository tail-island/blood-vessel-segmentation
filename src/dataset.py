import ctypes
import numpy as np
import os
import tensorflow as tf

from functools import reduce
from funcy import compose, juxt
from glob import glob
from math import ceil
from multiprocessing import Pool, Value, cpu_count
from operator import itemgetter
from sklearn.utils.extmath import cartesian


PATCH_WIDTH = 128
PATCH_HEIGHT = 128
PATCH_DEPTH = 32


# TODO: 見直す！　間違いが打ち消しあって間違えて動いている気がする。。。


def create_x_and_y(rng, volumetric_image_x, volumetric_image_y):
    def get_look_at(eye, center, up):
        z = eye - center
        z = z / np.linalg.norm(z)

        x = np.cross(up, z)
        x = x / np.linalg.norm(x)

        y = np.cross(z, x)
        y = y / np.linalg.norm(y)

        return np.array(((*x, np.dot(-eye, x)), (*y, np.dot(-eye, y)), (*z, np.dot(-eye, z)), (0, 0, 0, 1)))

    def get_scaling(scale):
        return np.array(((scale[0], 0, 0, 0), (0, scale[1], 0, 0), (0, 0, scale[2], 0), (0, 0, 0, 1)))

    def get_target_positions():
        result = cartesian((range(PATCH_WIDTH), range(PATCH_HEIGHT), range(PATCH_DEPTH))).T
        result = np.vstack((result, np.ones(np.shape(result)[1])))

        return result

    def get_source_positions(target_positions):
        z_max, y_max, x_max = np.shape(volumetric_image_x)

        eye = np.array((rng.uniform(0, x_max), rng.uniform(0, y_max), rng.uniform(0, z_max)))
        center = np.array((rng.uniform(0, x_max), rng.uniform(0, y_max), rng.uniform(0, z_max)))
        up = np.array((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
        up = up / np.linalg.norm(up)

        scale = 1  # 2 ** rng.normal(scale=0.1)

        result = np.matmul(np.linalg.inv(np.matmul(get_look_at(eye, center, up),
                                                   get_scaling(np.array((scale, scale, scale))))),
                           target_positions)

        return result

    while True:
        target_positions = (get_target_positions()).astype(np.int32)
        source_positions = (get_source_positions(target_positions) + 0.5).astype(np.int32)

        if np.any(source_positions < 0):
            continue

        target_indices = tuple(np.flip(target_positions[:3], axis=0))
        source_indices = tuple(np.flip(source_positions[:3], axis=0))

        try:
            x_values = volumetric_image_x[source_indices]
            y_values = volumetric_image_y[source_indices]
        except IndexError:
            continue

        x = np.zeros((PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH), dtype=np.float32)
        x[target_indices] = x_values
        x = np.expand_dims(x, 3)

        y = np.zeros((PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH), dtype=np.float32)
        y[target_indices] = y_values
        y = np.expand_dims(y, 3)

        return x, y


volumetric_image_x = None
volumetric_image_y = None


def create_and_save_data_initializer(volumetric_image_x_value, volumetric_image_x_shape, volumetric_image_y_value, volumetric_image_y_shape):
    global volumetric_image_x, volumetric_image_y

    volumetric_image_x = np.reshape(np.ctypeslib.as_array(volumetric_image_x_value.get_obj()), volumetric_image_x_shape)
    volumetric_image_y = np.reshape(np.ctypeslib.as_array(volumetric_image_y_value.get_obj()), volumetric_image_y_shape)


def create_and_save_data(rng, index):
    x, y = create_x_and_y(rng, volumetric_image_x, volumetric_image_y)

    np.savez(f"../data/dataset/{index:06}", x=x, y=y)


def create_and_save_dataset(rng, num_data_per_volumetric_image, volumetric_images_path):
    for data_path in glob('../data/dataset/*.npz'):
        os.remove(data_path)

    for i, (volumetric_image_x, volumetric_image_y) in enumerate(map(compose(juxt(itemgetter('x'), itemgetter('y')), np.load), sorted(glob(f"{volumetric_images_path}/*.npz")))):
        volumetric_image_x_value = Value(ctypes.c_float * reduce(lambda acc, dimension: acc * dimension, np.shape(volumetric_image_x), 1))
        np.ctypeslib.as_array(volumetric_image_x_value.get_obj())[:] = np.ravel(volumetric_image_x)

        volumetric_image_y_value = Value(ctypes.c_float * reduce(lambda acc, dimension: acc * dimension, np.shape(volumetric_image_y), 1))
        np.ctypeslib.as_array(volumetric_image_y_value.get_obj())[:] = np.ravel(volumetric_image_y)

        with Pool(cpu_count(), initializer=create_and_save_data_initializer, initargs=(volumetric_image_x_value, np.shape(volumetric_image_x), volumetric_image_y_value, np.shape(volumetric_image_y))) as p:
            p.starmap(create_and_save_data, map(lambda index: (np.random.default_rng(rng.integers(np.iinfo(np.int32).max)), index + i * num_data_per_volumetric_image), range(num_data_per_volumetric_image)))

        volumetric_image_x_value = volumetric_image_y_value = None
        volumetric_image_x = volumetric_image_y = None


class Sequence(tf.keras.utils.Sequence):
    def __init__(self, volumetric_images_path, batch_size, num_data_per_volumetric_image=10_000, dataset_life=10, seed=None):
        self.volumetric_images_path = volumetric_images_path
        self.batch_size = batch_size
        self.epoch = 0
        self.num_data_per_volumetric_image = num_data_per_volumetric_image
        self.dataset_life = dataset_life
        self.rng = np.random.default_rng(seed)

        os.makedirs('../data/dataset', exist_ok=True)

        self.on_epoch_end()

    def __len__(self):
        return ceil(len(self.data_paths) / self.batch_size)

    def __getitem__(self, index):
        index_begin = index * self.batch_size
        index_end = min(index_begin + self.batch_size, len(self.data_paths))

        return tuple(map(np.array, zip(*map(compose(juxt(itemgetter('x'), itemgetter('y')), np.load), self.data_paths[index_begin:index_end]))))

    def on_epoch_end(self):
        if self.epoch % self.dataset_life == 0:
            create_and_save_dataset(self.rng, self.num_data_per_volumetric_image, self.volumetric_images_path)
            self.data_paths = glob("../data/dataset/*.npz")

        self.epoch += 1
        self.rng.shuffle(self.data_paths)
