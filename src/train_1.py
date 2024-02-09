import tensorflow as tf

from dataset import PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, Sequence
from model import create_op
from util import Visualizer


DATASET_LIFE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200


sequence = Sequence('../data/volumetric_images/dense', batch_size=4, dataset_life=DATASET_LIFE)

inputs = tf.keras.Input(shape=(PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, 1))
op = create_op()

model = tf.keras.Model(inputs, op(inputs))
model.load_weights('../data/models/model-0/weights')

model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE), loss='binary_focal_crossentropy', metrics=('accuracy',))
model.fit(sequence, epochs=NUM_EPOCHS, callbacks=(Visualizer('model-1', epoch_interval=10),))
model.save_weights('../data/models/model-1/weights')
