# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import cv2
import os


from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCH = 20

root= '/data/test/'
TRAINPATH = 'train_v2'
train_names = os.listdir(root + TRAINPATH)
mask_csv = root + 'train_ship_segmentations_v2.csv'


masks = pd.read_csv(mask_csv)


_v_path_join = np.vectorize(os.path.join)
_v_file_size = np.vectorize(lambda fp: (os.stat(fp).st_size) / 1024)



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_train_data(masks):
    is_ship = (masks.groupby('ImageId')['EncodedPixels']
                   .count()
                   .reset_index()
                   .rename(columns={'EncodedPixels': 'ships'})
                   .assign(has_ship=lambda df: np.where(df['ships']>0, 1,0))
                   .assign(file_path=lambda df: _v_path_join(root + TRAINPATH, df.ImageId.astype(str)))
                   .assign(file_size_kb=lambda df: _v_file_size(df.file_path))
                   .loc[lambda df: df.file_size_kb > 50,  :])



    is_ship = is_ship.loc[lambda df: df.has_ship ==1]
    
    trains, vals = train_test_split(is_ship, test_size=0.3, stratify=is_ship['ships'])

    train_df = pd.merge(masks, trains)
    val_df = pd.merge(masks, vals)
    

    
    return train_df, val_df


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


train_df, val_df = make_train_data(masks)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

no_mask = np.zeros(IMG_WIDTH*IMG_HEIGHT, dtype=np.uint8)


#

def rle_decode(mask_rle, shape=(IMG_WIDTH, IMG_HEIGHT)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    if pd.isnull(mask_rle):
        img = no_mask
        return img.reshape(shape).T
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


#




def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def convert_mask_per_batch(input_info, batch_size, img_scaling=(1,1)):
    batches = list(input_info.groupby('ImageId'))
    batch_origin = []
    batch_mask = []
    
    while True:
        np.random.shuffle(batches)
        for image_name, mask in batches:
            image_path = os.path.join(root + TRAINPATH, image_name)
            image = cv2.imread(image_path)
            mask = masks_as_image(mask['EncodedPixels'].values)
            mask = np.expand_dims(mask, axis=-1)
            if img_scaling is not None:
                image = image[::img_scaling[0], ::img_scaling[1]]
                mask = mask[::img_scaling[0], ::img_scaling[1]]
            batch_origin += [image]
            batch_mask += [mask]
            if len(batch_origin) >= batch_size:
                yield np.stack(batch_origin, 0) / 255.0, np.stack(batch_mask, 0)
                batch_origin, batch_mask = [], []



#

                
                
def config_image_generator(apply_brightness=True):
    generator_parameter_dict = dict(featurewise_center=False,
                                   samplewise_center=False,
                                   rotation_range=45,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.01,
                                   zoom_range=[0.9, 1.25],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='reflect',
                                   data_format='channels_last')
    
    if apply_brightness:
        generator_parameter_dict['brightness_range'] = [0.5, 1.5]
    image_generator = ImageDataGenerator(**generator_parameter_dict)
    
    if apply_brightness:
        generator_parameter_dict.pop('brightness_range')
    mask_generator = ImageDataGenerator(**generator_parameter_dict)
    
    return image_generator, mask_generator



#


def augmentation_generator(input_generator, apply_brightness=True, seed=None):
    image_generator, mask_generator = config_image_generator(apply_brightness)
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    
    for image, mask in input_generator:
        seed = np.random.choice(range(9999))
        gen_image = image_generator.flow(255 * image,
                                        batch_size=image.shape[0],
                                        seed=seed,
                                        shuffle=True)
        gen_mask = mask_generator.flow(mask,
                                      batch_size=image.shape[0],
                                      seed=seed,
                                      shuffle=True)
        
        yield next(gen_image) / 255.0, next(gen_mask)
    

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def unet(input_size = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)):
    inputs = Input(shape=(input_size))
    conv0 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv0 = BatchNormalization()(conv0)
    conv0 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0)
    conv0 = BatchNormalization()(conv0)

    comp0 = AveragePooling2D((6,6))(conv0)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(comp0)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.4)(conv1)

    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.4)(conv2)

    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.4)(conv3)

    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.4)(conv4)

    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    upcv6 = UpSampling2D(size=(2,2))(conv5)
    upcv6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv6)
    upcv6 = BatchNormalization()(upcv6)
    mrge6 = concatenate([conv4, upcv6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    upcv7 = UpSampling2D(size=(2,2))(conv6)
    upcv7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv7)
    upcv7 = BatchNormalization()(upcv7)
    mrge7 = concatenate([conv3, upcv7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    upcv8 = UpSampling2D(size=(2,2))(conv7)
    upcv8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv8)
    upcv8 = BatchNormalization()(upcv8)
    mrge8 = concatenate([conv2, upcv8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    upcv9 = UpSampling2D(size=(2,2))(conv8)
    upcv9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv9)
    upcv9 = BatchNormalization()(upcv9)
    mrge9 = concatenate([conv1, upcv9], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    dcmp10 = UpSampling2D((6,6), interpolation='bilinear')(conv9)
    mrge10 = concatenate([dcmp10, conv0], axis=3)
    conv10 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv10)


    model = Model(inputs, outputs=conv11)

    return model

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#steps_per_epochs = len(train_df) // BATCH_SIZE
steps_per_epochs = 200

train_generator = augmentation_generator(convert_mask_per_batch(train_df, BATCH_SIZE))
val_generator = augmentation_generator(convert_mask_per_batch(val_df, BATCH_SIZE))

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------




def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

model = unet()


model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])

   
model.summary()


checkpoint = ModelCheckpoint('weights.h5', monitor='val_dice_coef', verbose=0, save_best_only=True, mode='max', save_weights_only=True)
earlystop = EarlyStopping(monitor='val_dice_coef', mode='max', patience=15)
reducelr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3, verbose=0, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)

CALLBACKS = [checkpoint, earlystop, reducelr]




model.fit_generator(train_generator, steps_per_epoch=steps_per_epochs,
                          epochs=EPOCH, validation_data=val_generator,
                          callbacks=CALLBACKS, validation_steps=steps_per_epochs)



model.save_weights('final_model.h5')
