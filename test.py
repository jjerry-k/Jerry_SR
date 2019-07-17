from layer import *
from tensorflow.keras import layers, models

x = layers.Input(shape=(28, 28, 3, ))

block1 = res_block_original(x, 4, name='block_1')

block2 = res_block_original(block1, 4, identity=False, name='block_2')

model = models.Model(inputs=x, outputs=block2)

model.summary()