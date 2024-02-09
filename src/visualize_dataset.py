import matplotlib.animation as animation
import matplotlib.pyplot as plot
import numpy as np

from dataset import PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, Sequence


sequence = Sequence('../data/volumetric_images/all/', batch_size=10, num_data_per_volumetric_image=10, seed=1234)

for i in range(3):
    xs, ys = sequence[i]

    for x, y in zip(xs, ys):
        x = np.reshape((x * 255).astype(np.uint8), (PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH))
        y = np.reshape((y * 255).astype(np.uint8), (PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH))

        figure = plot.figure(figsize=(16, 9))

        subplot_1 = figure.add_subplot(1, 2, 1)
        subplot_2 = figure.add_subplot(1, 2, 2)

        images = []

        for depth in range(np.shape(x)[0]):
            x_image = x[depth, :, :]
            x_image = np.stack((x_image, x_image, x_image), axis=-1)

            y_image = y[depth, :, :]
            y_image = np.stack((y_image, y_image, y_image), axis=-1)

            images.append((subplot_1.imshow(x_image), subplot_2.imshow(y_image)))

        artist_animation = animation.ArtistAnimation(figure, images, interval=100, repeat_delay=1_000)
        plot.show()
