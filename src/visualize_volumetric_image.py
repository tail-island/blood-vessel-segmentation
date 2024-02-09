import matplotlib.animation as animation
import matplotlib.pyplot as plot
import numpy as np

from funcy import juxt
from glob import glob
from operator import itemgetter


for path in sorted(glob('../data/volumetric_images/all/*.npz')):
    x, y = juxt(itemgetter('x'), itemgetter('y'))(np.load(path))

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
