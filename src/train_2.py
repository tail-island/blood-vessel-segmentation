import tensorflow as tf

from dataset import PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, Sequence
from model import create_op
from util import Visualizer


BATCH_SIZE = 4
NUM_EPOCHS = 100


sequence = Sequence('../data/volumetric_images/dense', batch_size=BATCH_SIZE, dataset_life=1)

inputs = tf.keras.Input(shape=(PATCH_DEPTH, PATCH_HEIGHT, PATCH_WIDTH, 1))
op = create_op()

model = tf.keras.Model(inputs, op(inputs))
model.load_weights('../data/models/model-1/weights')

model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.00025,
                                                                                                          decay_steps=NUM_EPOCHS * len(sequence))),
              loss='binary_focal_crossentropy',
              metrics=('accuracy',))
model.fit(sequence,
          epochs=NUM_EPOCHS,
          callbacks=(Visualizer('model-2', epoch_interval=10),))
model.save_weights('../data/models/model-2/weights')
