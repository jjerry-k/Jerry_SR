import os
import numpy as np
import nibabel as nib
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, DenseNet121

def load_nii(PATH):
    """
    Input
    PATH : Path of Nifti file
    
    Output
    output : Image of Nifti [Height, Width, Slices]
    """
    data = nib.load(PATH)
    img = data.get_data()
    img = np.rot90(img)
    
    return img

def load_nii_multi(PAT_PATH, img = None):
    """
    Input
    PATH : Path of patient
    
    Output
    t1_img : Image of patient's T1
    dante_img : Image of patient's DANTE
    """
    SEQ_LISTS = ['t1setrafs', 'T1SPACE09mmISOPOSTwDANTE']
    t1_PATH = os.path.join(PAT_PATH, SEQ_LISTS[0])
    t1_img_list = [img for img in os.listdir(t1_PATH) if '.nii' in img]
    #print(t1_img_list)
    t1_img_PATH = os.path.join(t1_PATH, t1_img_list[0])

    dante_PATH = os.path.join(PAT_PATH, SEQ_LISTS[1])
    dante_img_list = [img for img in os.listdir(dante_PATH) if '.nii' in img]
    #print(dante_img_list)
    dante_img_PATH = os.path.join(dante_PATH, dante_img_list[0])

    if img == 'rsl':
        t1_img_PATH = os.path.join(t1_PATH, t1_img_list[1])
        dante_img_PATH = os.path.join(dante_PATH, dante_img_list[1])
        
        t1_img = load_nii(t1_img_PATH)
        dante_img = load_nii(dante_img_PATH)
        
        return t1_img[:,2:-2,:], dante_img[:,2:-2,:]
    
    else :
        t1_img = load_nii(t1_img_PATH)
        dante_img = load_nii(dante_img_PATH)
        #print(dante_img.shape)
        n_slice = dante_img.shape[-1]
        low_slices = n_slice//6
        high_slices = low_slices*6
        diff = n_slice-high_slices
        half = diff//2
        #print(diff, half)
        if diff == 0:
            return t1_img[:,2:-2,:], dante_img
        else:
            if diff % 2 == 0:
                dante_img = dante_img[..., half:-half]
            else :
                dante_img = dante_img[..., half:-(half+1)]
            #print(dante_img.shape)    
            return t1_img[:,2:-2,:], dante_img

def data_loader_v1(PATH, val_idx = 0, img = 'rsl'):
    train_t1 = None
    train_dante = None
    Pat_lists = sorted(os.listdir(PATH))
    val_pat = Pat_lists[val_idx]
    Pat_lists.pop(val_idx)
    for i, pat in enumerate(Pat_lists):
        Pat_path = os.path.join(PATH, pat)
        tmp_t1, tmp_dante = load_nii_multi(Pat_path, img = img)
        tmp_dante = np.transpose(tmp_dante, [2, 0, 1])
        tmp_dante = np.reshape(tmp_dante, [len(tmp_dante)//6, 6, 320, 256])
        tmp_dante = np.transpose(tmp_dante, [0, 2, 3, 1])
        tmp_t1 = np.expand_dims(np.transpose(tmp_t1, [2, 0, 1]), -1)
        if i==0:
            train_t1 = tmp_t1[2:]
            train_dante = tmp_dante[2:]
        else:
            train_t1 = np.concatenate([train_t1, tmp_t1[2:]], axis=0)
            train_dante = np.concatenate([train_dante, tmp_dante[2:]], axis=0)
    
    Pat_path = os.path.join(PATH, val_pat)
    val_t1, val_dante = load_nii_multi(Pat_path,  img = img)
    val_dante = np.transpose(val_dante, [2, 0, 1])
    val_dante = np.reshape(val_dante, [len(val_dante)//6, 6, 320, 256])
    val_dante = np.transpose(val_dante, [0, 2, 3, 1])
    val_t1 = np.expand_dims(np.transpose(val_t1, [2, 0, 1]), -1)
    
    return train_t1, train_dante, val_t1[2:], val_dante[2:]

def data_loader_v2(PATH, val_idx = 0, img = 'rsl', norm=False):
    train_low = []
    train_high = None
    
    val_low = []
    
    test_low = []
    
    Pat_lists = sorted(os.listdir(PATH))
    val_pat = Pat_lists[val_idx]
    Pat_lists.pop(val_idx)
    for i, pat in enumerate(Pat_lists):
        Pat_path = os.path.join(PATH, pat)
        tmp_t1, tmp_dante = load_nii_multi(Pat_path,  img = img) # (H, W, S)
        if norm:
            tmp_t1 = tmp_t1/tmp_t1.max()
            tmp_dante = tmp_dante/tmp_dante.max()
        h, w, s = tmp_dante.shape
        
        tmp_dante = np.array(np.dsplit(tmp_dante, s/6)) # (S/6, H, W, 6)
        
        # Make Train Low
        mean_dante = tmp_dante.mean(axis=-1)  # (S/6, H, W)
        for j in range(s//6):
            empty = np.zeros((3, h, w))
            if j==0:
                continue#empty[1:,...] = mean_dante[:2]
            elif j==s/6-1:
                empty[:2,...] = mean_dante[-2:]
            else:
                empty = mean_dante[j-1:j+2]
            
            empty = np.transpose(empty, [1, 2, 0])
            train_low.append(empty)
        
        
        tmp_t1 = np.transpose(tmp_t1, (2, 0, 1)) # (S, H, W)
        for j in range(s//6):
            empty = np.zeros((3, h, w))
            if j==0:
                continue#empty[1:,...] = tmp_t1[:2]
            elif j==s/6-1:
                empty[:2,...] = tmp_t1[-2:]
            else:
                empty = tmp_t1[j-1:j+2]
            
            empty = np.transpose(empty, [1, 2, 0])
            test_low.append(empty)
            
        if i==0:
            train_high = tmp_dante[1:]
        else:
            train_high = np.concatenate([train_high, tmp_dante[1:]], axis=0)
    
    Pat_path = os.path.join(PATH, val_pat)
    test_t1, val_high = load_nii_multi(Pat_path,  img = img)
    if norm:
        test_t1 = test_t1/test_t1.max()
        val_high = val_high/val_high.max()
    h, w, s = val_high.shape
    val_high = np.array(np.dsplit(val_high, s/6))
    mean_dante = val_high.mean(axis=-1)  # (S/6, H, W)
    for j in range(s//6):
        empty = np.zeros((3, h, w))
        if j==0:
            continue#empty[1:,...] = mean_dante[:2]
        elif j==s/6-1:
            empty[:2,...] = mean_dante[-2:]
        else:
            empty = mean_dante[j-1:j+2]

        empty = np.transpose(empty, [1, 2, 0])
        val_low.append(empty)


    test_t1 = np.transpose(test_t1, (2, 0, 1)) # (S, H, W)
    for j in range(s//6):
        empty = np.zeros((3, h, w))
        if j==0:
            continue#empty[1:,...] = test_t1[:2]
        elif j==s/6-1:
            empty[:2,...] = test_t1[-2:]
        else:
            empty = test_t1[j-1:j+2]

        empty = np.transpose(empty, [1, 2, 0])
        test_low.append(empty)
        
    #val_dante = np.transpose(val_dante, [2, 0, 1])
    #val_dante = np.reshape(val_dante, [len(val_dante)//6, 6, 320, 256])
    #val_dante = np.transpose(val_dante, [0, 2, 3, 1])
    #val_t1 = np.expand_dims(np.transpose(val_t1, [2, 0, 1]), -1)
    
    return np.array(train_low), train_high, np.array(val_low), val_high[1:], np.array(test_low)


def data_loader_v3(PATH, val_idx = 0, img = 'rsl', norm=False):
    train_low = []
    train_high = None
    val_low = []
    test_low = []
    sequence = {'train':[], 'test':[]}
    # Loading Patient list & Seperating Train lists & Validation lists
    Pat_lists = sorted(os.listdir(PATH))
    val_pat = Pat_lists[val_idx]
    Pat_lists.pop(val_idx)
    
    
    # Train lists loop
    for i, pat in enumerate(Pat_lists):
        # Load t1, dante
        Pat_path = os.path.join(PATH, pat)
        #print(Pat_path)
        tmp_t1, tmp_dante = load_nii_multi(Pat_path,  img = img) # (H, W, S)
        if norm:
            tmp_t1 = tmp_t1/tmp_t1.max()
            tmp_dante = tmp_dante/tmp_dante.max()
        # Spliting dante data (H, W, S) -> (S/6, H, W, 6)
        _, _, s = tmp_dante.shape
        #print(tmp_dante.shape)
        tmp_dante = np.array(np.dsplit(tmp_dante, s/6))[1:] # 첫번째 data는 불량으로 제외.
        batch, h, w, _ = tmp_dante.shape
        #sequence['train'].append(len(batch))
        # Make Train Low
        mean_dante = tmp_dante.mean(axis=-1)  # (S/6, H, W)
        for j in range(batch):
            empty = np.zeros((3, h, w))
            if j==0:
                empty[1:,...] = mean_dante[:2]
            elif j==batch-1:
                empty[:2,...] = mean_dante[-2:]
            else:
                empty = mean_dante[j-1:j+2]
            
            empty = np.transpose(empty, [1, 2, 0])
            train_low.append(empty)
        
        # Make Train High
        tmp_high = np.zeros((batch, h, w, 12))
        tmp_high[..., 3:-3] = tmp_dante
        tmp_high[1:, ..., :3] = tmp_dante[:-1, ..., 3:]
        tmp_high[:-1, ..., -3:] = tmp_dante[1:, ..., :3]
        
        if i==0:
            train_high = tmp_high
        else:
            train_high = np.concatenate([train_high, tmp_high], axis=0)
            
        # Make Test Low
        tmp_t1 = np.transpose(tmp_t1, (2, 0, 1)) # (S, H, W)
        batch, h, w = tmp_t1.shape
        for j in range(batch):
            empty = np.zeros((3, h, w))
            if j==0:
                empty[1:,...] = tmp_t1[:2]
            elif j==batch-1:
                empty[:2,...] = tmp_t1[-2:]
            else:
                empty = tmp_t1[j-1:j+2]
            
            empty = np.transpose(empty, [1, 2, 0])
            test_low.append(empty)
        #sequence['test'].append(batch)
    # Validation
    
    # Load image
    Pat_path = os.path.join(PATH, val_pat)
    test_t1, tmp_val_high = load_nii_multi(Pat_path,  img = img)
    if norm:
        test_t1 = test_t1/test_t1.max()
        tmp_val_high = tmp_val_high/tmp_val_high.max()
    # Spliting dante data (H, W, S) -> (S/6, H, W, 6)
    h, w, s = tmp_val_high.shape
    tmp_val_high = np.array(np.dsplit(tmp_val_high, s/6))[1:]
    batch, h, w, _ = tmp_val_high.shape
    
    # Make Validation Low
    mean_dante = tmp_val_high.mean(axis=-1)  # (S/6, H, W)
    
    for j in range(batch):
        empty = np.zeros((3, h, w))
        if j==0:
            empty[1:,...] = mean_dante[:2]
        elif j==batch-1:
            empty[:2,...] = mean_dante[-2:]
        else:
            empty = mean_dante[j-1:j+2]

        empty = np.transpose(empty, [1, 2, 0])
        val_low.append(empty)

    # Make Train High
    val_high = np.zeros((batch, h, w, 12))
    val_high[..., 3:-3] = tmp_val_high
    val_high[1:, ..., :3] = tmp_val_high[:-1, ..., 3:]
    val_high[:-1, ..., -3:] = tmp_val_high[1:, ..., :3]
    
    
    
    
    test_t1 = np.transpose(test_t1, (2, 0, 1)) # (S, H, W)
    s, h, w = test_t1.shape
    for j in range(s//6):
        empty = np.zeros((3, h, w))
        if j==0:
            empty[1:,...] = test_t1[:2]
        elif j==s/6-1:
            empty[:2,...] = test_t1[-2:]
        else:
            empty = test_t1[j-1:j+2]

        empty = np.transpose(empty, [1, 2, 0])
        test_low.append(empty)
        
    #val_dante = np.transpose(val_dante, [2, 0, 1])
    #val_dante = np.reshape(val_dante, [len(val_dante)//6, 6, 320, 256])
    #val_dante = np.transpose(val_dante, [0, 2, 3, 1])
    #val_t1 = np.expand_dims(np.transpose(val_t1, [2, 0, 1]), -1)
    
    return np.array(train_low), train_high, np.array(val_low), val_high, np.array(test_low)




def load_base_model(backbone = 'vgg', **params):
    '''
    Loading backbone network
    ====== Input ======
    backbone : Network name of backbone 
        ['vgg', 'resnet', 'xception', 'mobile', 'dense']
    params : "include_top", "weights", "input_tensor", "input_shape", "pooling", "classes"
    
    ====== Output ======
    base_model : Keras Model instance
    '''
    model_dict = {'vgg':'VGG16',
                 'resnet':'ResNet50',
                 'xception':'Xception',
                 'mobile':'MobileNet',
                 'IRv2':'InceptionResNetV2',
                 'Iv3':'InceptionV3',
                 'dense':'DenseNet121',
                 'nas':'NASNetMobile'}
    
    param_dict = {"include_top" : False, 
                  "weights" : None,
                  "input_tensor" : None,
                  "input_shape" : None, 
                  "pooling" : None, 
                  "classes" : 1000}
    
    if params :
        for key in params.keys():
            if key  in ['weights', 'pooling']:
                param_dict[key] = "'%s'"%(params[key])
            else : 
                param_dict[key] = params[key]
    
    extract = lambda a: ''.join(i+', ' for i in [i+'='+str(a[i]) for i in a])
    to_cmd = extract(param_dict)
    command = "base_model = %s(%s)"%(model_dict[backbone], to_cmd)
    print("Loading %s model"%model_dict[backbone])
    #print(command)
    exec(command, globals())
    non_train_params = [layer.shape.num_elements() for layer in base_model.non_trainable_weights]
    non_train_params = sum(non_train_params)
    print("Total Parameter of Model : ", format(base_model.count_params(), ','))
    print("Trainable Parameter of Model : ", format(base_model.count_params()-non_train_params, ','))
    print("Non-Trainable Parameter of Model : ", format(non_train_params, ','))
    return base_model





def Xception(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the Xception architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Note that the default input image size for this model is 299x299.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True,
            and if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    
    x = layers.ZeroPadding2D((1, 1))(img_input)
    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='xception')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels.h5',
                TF_WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
        else:
            weights_path = keras_utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                TF_WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='b0042744bf5b25fce3cb969f33bebb97')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)