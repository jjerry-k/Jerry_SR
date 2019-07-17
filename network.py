from tensorflow.keras import layers, models, activations, optimizers
import sys
sys.path.append(['.','..'])
from utils import *
from layer import *

def generator(model = 'vgg', n_slice=6, case=1):
    '''
    model : 'vgg', 'resnet', 'xception', 'mobile', 'dense'
    '''
    block_dict = {
        "vgg" : ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'],
        "resnet" : ['activation_40', 'activation_22', 'activation_10', 'activation_1'],
        "xception" : ['block13_sepconv2_bn', 'block4_sepconv2_bn', 'block3_sepconv2_bn', 'block1_conv1_act'],
        "mobile" : ['conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'], 
        "dense" : ['pool4_conv', 'pool3_conv', 'pool2_conv', 'conv1/relu']
    }
    # ========= Encoder ==========
    print("=========== Information about Backbone ===========")
    base_model = load_base_model(model, input_shape=(None, None, 3))
    x = layers.Conv2D(1024, 3, padding='same', activation='relu')(base_model.output) # H/32

    # ========= Decoder ==========
    x = layers.UpSampling2D(interpolation='bilinear')(x) # H/16
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][0]).output], axis = -1)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(interpolation='bilinear')(x) # H/8
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][1]).output], axis = -1)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(interpolation='bilinear')(x) # H/4
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][2]).output], axis = -1)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(interpolation='bilinear')(x) # H/2
    x = layers.concatenate([x, base_model.get_layer(block_dict[model][3]).output], axis = -1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(interpolation='bilinear')(x) # H
    if model == 'vgg':
        x = layers.concatenate([x, base_model.get_layer(block_dict[model][4]).output], axis = -1)
    output = layers.Conv2D(n_slice, 3, padding='same', activation='relu')(x)

    #output = layers.DepthwiseConv2D(3, padding='same')(x)
    if case == 2:
        Net = models.Model(base_model.input, output)
    else : 
        Net = models.Model(base_model.input, [output, output])
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net

def discriminator(model = 'vgg', n_slice=6, case=1):
    '''
    model : 'vgg', 'resnet', 'xception', 'mobile', 'dense'
    '''
    block_dict = {
        "vgg" : ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'],
        "resnet" : ['activation_40', 'activation_22', 'activation_10', 'activation_1'],
        "xception" : ['block13_sepconv2_bn', 'block4_sepconv2_bn', 'block3_sepconv2_bn', 'block1_conv1_act'],
        "mobile" : ['conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'], 
        "dense" : ['pool4_conv', 'pool3_conv', 'pool2_conv', 'conv1/relu']
    }
    # ========= Extractor ==========
    print("=========== Information about Backbone ===========")
    base_model = load_base_model(model, input_shape=(None, None, n_slice))
    texture = layers.Conv2D(1024, 3, padding='same', activation='relu')(base_model.output) # H/32
    
    # ========= Classifier ==========
    x = layers.GlobalAvgPool2D()(texture)
    output = layers.Dense(1, activation='sigmoid')(x)
    if case==2:
        Net = models.Model(base_model.input, [output, texture])
    else:
        Net = models.Model(base_model.input, output)
    
    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net


def unet(n_slice=6, case=1):
    x = layers.Input(shape=(None, None, 3))
    block1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv1')(x)
    block1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block1_conv2')(block1)

    pool1 = layers.Conv2D(64, 2, strides=(2, 2), name='block1_conv3')(block1)

    block2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv1')(pool1)
    block2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block2_conv2')(block2)

    pool2 = layers.Conv2D(128, 2, strides=(2, 2), name='block2_conv3')(block2)

    block3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv1')(pool2)
    block3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv2')(block3)
    block3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block3_conv3')(block3)

    pool3 = layers.Conv2D(256, 2, strides=(2, 2), name='block3_conv4')(block3)

    block4 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv1')(pool3)
    block4 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv2')(block4)
    block4 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block4_conv3')(block4)

    pool4 = layers.Conv2D(512, 2, strides=(2, 2), name='block4_conv4')(block4)

    block5 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv1')(pool4)
    block5 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv2')(block5)
    block5 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block5_conv3')(block5)

    unpool1 = layers.Conv2DTranspose(512, 4, strides=(2, 2), padding='same')(block5)
    concat1 = layers.Concatenate(axis = 3)([unpool1, block4])
    block6 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block6_conv1')(concat1)
    block6 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block6_conv2')(block6)
    block6 = layers.Conv2D(512, 3, padding='same', activation='relu', name='block6_conv3')(block6)

    unpool2 = layers.Conv2DTranspose(256, 4, strides=(2, 2), padding='same')(block6)
    concat2 = layers.Concatenate(axis = 3)([unpool2, block3])
    block7 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block7_conv1')(concat2)
    block7 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block7_conv2')(block7)
    block7 = layers.Conv2D(256, 3, padding='same', activation='relu', name='block7_conv3')(block7)

    unpool3 = layers.Conv2DTranspose(128, 4, strides=(2, 2), padding='same')(block7)
    concat3 = layers.Concatenate(axis = 3)([unpool3, block2])
    block8 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block8_conv1')(concat3)
    block8 = layers.Conv2D(128, 3, padding='same', activation='relu', name='block8_conv2')(block8)

    unpool4 = layers.Conv2DTranspose(128, 4, strides=(2, 2), padding='same')(block8)
    concat4 = layers.Concatenate(axis = 3)([unpool4, block1])
    block9 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block9_conv1')(concat4)
    block9 = layers.Conv2D(64, 3, padding='same', activation='relu', name='block9_conv2')(block9)

    output = layers.Conv2D(n_slice, 3, padding='same', activation='relu', name='output')(block9)

    if case == 2:
        Net = models.Model(x, output)
    else :
        Net = models.Model(x, [output, output])

    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net

def res_unet(n_slice=6, case=1):
    x = layers.Input(shape=(None, None, 3))
    block1 = res_block_original(x, 16, name='block_1_1')
    block1 = res_block_original(block1, 16, identity=False, name='block_1_2')

    block2 = res_block_original(block1, 32, stride=2, name='block_2_1')
    block2 = res_block_original(block2, 32, identity=False, name='block_2_2')
    block2 = res_block_original(block2, 32, identity=False, name='block_2_3')

    block3 = res_block_original(block2, 64, stride=2, name='block_3_1')
    block3 = res_block_original(block3, 64, identity=False, name='block_3_2')
    block3 = res_block_original(block3, 64, identity=False, name='block_3_3')

    block4 = res_block_original(block3, 128, stride=2, name='block_4_1')
    block4 = res_block_original(block4, 128, identity=False, name='block_4_2')
    block4 = res_block_original(block4, 128, identity=False, name='block_4_3')

    block5 = res_block_original(block4, 256, stride=2, name='block_5_1')
    block5 = res_block_original(block5, 256, identity=False, name='block_5_2')
    block5 = res_block_original(block5, 256, identity=False, name='block_5_3')

    up_block4 = layers.Conv2DTranspose(512, 4, strides=(2,2), padding='same', name='up_block4_transpose')(block5)
    up_block4 = layers.Concatenate(axis=3, name='up_block4_concat')([block4, up_block4])
    up_block4 = res_block_original(up_block4, 128,  name='up_block4')
    up_block4 = res_block_original(up_block4, 128, identity=False, name='up_block4')

    up_block3 = layers.Conv2DTranspose(256, 4, strides=(2,2), padding='same', name='up_block3_transpose')(up_block4)
    up_block3 = layers.Concatenate(axis=3, name='up_block3_concat')([block3, up_block3])
    up_block3 = res_block_original(up_block3, 64,  name='up_block3')
    up_block3 = res_block_original(up_block3, 64, identity=False, name='up_block3')

    up_block2 = layers.Conv2DTranspose(128, 4, strides=(2,2), padding='same', name='up_block2_transpose')(up_block3)
    up_block2 = layers.Concatenate(axis=3, name='up_block2_concat')([block2, up_block2])
    up_block2 = res_block_original(up_block2, 32, name='up_block2')
    up_block2 = res_block_original(up_block2, 32, identity=False, name='up_block2')

    up_block1 = layers.Conv2DTranspose(64, 4, strides=(2,2), padding='same', name='up_block1_transpose')(up_block2)
    up_block1 = layers.Concatenate(axis=3, name='up_block1_concat')([block1, up_block1])
    up_block1 = res_block_original(up_block1, 16, name='up_block1')
    up_block1 = res_block_original(up_block1, 16, identity=False, name='up_block1')

    output = layers.Conv2D(n_slice, 3, padding='same', activation='relu', name='prediction')(up_block1)
    
    if case == 2:
        Net = models.Model(x, output)
    else :
        Net = models.Model(x, [output, output])

    non_train_params = [layer.shape.num_elements() for layer in Net.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("\n=========== Information about Whole Network ===========")
    print("Total Parameter of Model : ", format(Net.count_params(), ','))
    print("Trainable Parameter of Model : ", format(Net.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return Net