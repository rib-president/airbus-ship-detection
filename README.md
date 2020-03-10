
# **Airbus-ship-Detection from satellite pictures**


## Table of Contents  
* [About](#about)
* [How to](#how-to)
* [Architecture](#architecture)
* [Result](#result)
  
   
  

## About
바다 위성사진에서 수송선을 찾아 segmentation하는 프로젝트입니다. 
    
  
   
제공된 **학습 데이터**는 바다 위성사진과  [이미지 파일명, EncodedPixels]로 이루어진 csv 파일입니다.  
  
*<EncodedPixels: 수송선의 위치를 (start, length)로 표현, (10, 3) -> (10, 12)>*
  
  
|ImageId|EncodedPixels|
|:---:|:---:|
|fffcb6464.jpg|222782 4 223550 10 224318 14 225086 14 225854 ...|
|fffcd0cf7.jpg|NaN|
|fffcda6d6.jpg|NaN|
|fffd924fb.jpg|110011 5 110779 5 111547 5 112315 5 113083 5 ...|
|fffd924fb.jpg|198440 8 199208 8 199976 8 200744 8 201512 8 ...|  
  
  
EncodedPixels을 Run-Length Decode 하여 얻은 이미지의 mask로 segmentation 모델 학습  
   
**RLE Decode**: *반복되는 값이 나타나는 지점의 pixel과 개수로 encode된 것을 다시 원래 이미지로 decode*
  
  
  
## How to
### 1. Data Preprocessing
* 수송선이 없고 바다 배경만 있는 이미지, 50kb 이하의 low resolution 이미지 제외  
    
  
``` python
is_ship = (masks.groupby('ImageId')['EncodedPixels']
               .count()
               .reset_index()
               .rename(columns={'EncodedPixels': 'ships'})
               .assign(has_ship=lambda df: np.where(df['ships']>0, 1,0))
               .assign(file_path=lambda df: _v_path_join(root + TRAINPATH, df.ImageId.astype(str)))
               .assign(file_size_kb=lambda df: _v_file_size(df.file_path))
               .loc[lambda df: df.file_size_kb > 50,  :])

is_ship = is_ship.loc[lambda df: df.has_ship ==1]
```
![](/sample/masks3.JPG)
  


  
* RLE decode로 학습에 필요한 mask 획득
  

```python
starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

starts -= 1
ends = starts + lengths
img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

for lo, hi in zip(starts, ends):
    img[lo:hi] = 1
```
![](/sample/plt1.JPG)
   
  
  
  
  
> augmentation은 학습 단계에서 keras ImageDataGenerator로 진행(brightness, flip, rotate, zoom, shear, shift)
  
``` python
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
```
![](/sample/aug.jpg)
  
  
  
  
  
### 2. Modeling
U-Net으로 구현  
  
  
![](/sample/unet.jpg)
  


``` python
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
```
  
  
  
  
    
  
  
## Architecture  
  
|Stages|Preprocessing|Modeling|
|:----:|:-----------:|:------:|
|Libraries|pandas, keras|keras|
Language|Python|Python|
  
  
## Result
![](/sample/shipres1.jpg)
  
  

![](/sample/shipres2.jpg)
  

![](/sample/shipres3.jpg)
    
   
  