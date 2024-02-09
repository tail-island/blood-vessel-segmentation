import tensorflow as tf

from dataset import PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, Sequence
from funcy import identity, juxt
from model import create_op
from util import Visualizer


DATASET_LIFE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 700


sequence = Sequence('../data/volumetric_images/all', batch_size=4, dataset_life=DATASET_LIFE)

model = tf.keras.Model(*juxt(identity, create_op())(tf.keras.Input(shape=(PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, 1))))
model.summary()

model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=LEARNING_RATE,
                                                                                                                  first_decay_steps=100 * len(sequence),)),
              loss='binary_focal_crossentropy',
              metrics=('accuracy',))
model.fit(sequence, epochs=NUM_EPOCHS, callbacks=(Visualizer('model-0', epoch_interval=10),))
model.save_weights('../data/models/model-0/weights')
