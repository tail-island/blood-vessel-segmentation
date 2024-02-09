import matplotlib.animation as animation
import matplotlib.pyplot as plot
import numpy as np
import os
import tensorflow as tf

from glob import glob
from pathlib import Path
from dataset import PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH


class Visualizer(tf.keras.callbacks.Callback):
    def __init__(self, prefix, epoch_interval=1):
        self.prefix = prefix
        self.epoch_interval = epoch_interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_interval != 0:
            return

        os.makedirs(f"../data/temp/{self.prefix}/{epoch + 1:04}", exist_ok=True)

        for path in map(Path, sorted(glob('../data/sample-data/*.npz'))):
            data = np.load(str(path))

            x = data['x']
            y_true = data['y']

            y_pred = self.model.predict_on_batch(np.reshape(x, (1, PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, 1)))

            x      = np.reshape((x      * 255).astype(np.uint8), (PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH))  # noqa: E221
            y_true = np.reshape((y_true * 255).astype(np.uint8), (PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH))
            y_pred = np.reshape((y_pred * 255).astype(np.uint8), (PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH))

            figure = plot.figure(figsize=(16, 9))

            subplot_1 = figure.add_subplot(1, 3, 1)
            subplot_2 = figure.add_subplot(1, 3, 2)
            subplot_3 = figure.add_subplot(1, 3, 3)

            images = []

            for depth in range(np.shape(x)[0]):
                x_image = x[depth, :, :]
                x_image = np.stack((x_image, x_image, x_image), axis=-1)

                y_true_image = y_true[depth, :, :]
                y_true_image = np.stack((y_true_image, y_true_image, y_true_image), axis=2)

                y_pred_image = y_pred[depth, :, :]
                y_pred_image = np.stack((y_pred_image, y_pred_image, y_pred_image), axis=2)

                images.append((subplot_1.imshow(x_image), subplot_2.imshow(y_true_image), subplot_3.imshow(y_pred_image)))

            artist_animation = animation.ArtistAnimation(figure, images, interval=100, repeat_delay=1000)
            artist_animation.save(f"../data/temp/{self.prefix}/{epoch + 1:04}/{path.stem}.mp4", writer='ffmpeg')

            plot.close()
