import os
import numpy as np

from dataset import create_x_and_y
from funcy import compose, juxt
from glob import glob
from operator import itemgetter


rng = np.random.default_rng(1234)

os.makedirs('../data/sample-data', exist_ok=True)

index = 0
for volumetric_image_x, volumetric_image_y in map(compose(juxt(itemgetter('x'), itemgetter('y')), np.load), sorted(glob('../data/volumetric_images/dense/*.npz'))):
    for _ in range(10):
        x, y = create_x_and_y(rng, volumetric_image_x, volumetric_image_y)
        np.savez(f"../data/sample-data/{index:06}", x=x, y=y)

        index += 1

    volumetric_image_x = volumetric_image_y = None
