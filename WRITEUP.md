# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

## Code organization

The project's main codebase reside in two Python modules, `bclone.py` and `model.py`. The former comprises a collection of functions for data loading, data generators preparation, and model training. The latter includes definition of the neural network architecture and routines for data preparation and launching the training procedure.

After each epoch is finished, the corresponding intermediate model is saved to disk.


## Data gathering

Data used for training the models is comprised of several data sets:

* the original training data provided by the Udacity staff (`STD`)
* data gathered by me using the simulator (`MYDRIVE_{1-4}`)
* "special cases" data gathered by me using the simulator (`SPECIAL`, `SPECIAL2`)

While running numerous experiments with different data sets and network architectures, it was observed that specifics of the data have the most significant impacts on the resulting driving behavior. In particular, the "special cases" examples were important to tackle the difficult zones, but they had to be balanced with the "normal driving" examples. As a rule of thumb, the "normal driving" data has to prevail.

## Model architecture

The selected neural network architecture is based on the Nvidia's architecture proposed [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), with modifications to accommodate for dimensions of images used in this project.

The network operated on (160, 320, 3) images, captured by the driving simulator. An input image is cropped to retain only the road region, which is further fed into a series of convolutional layers. The final convolutional layer is flattened and fed into a series of fully-connected layers. Both convolutional and fully-connected layers include dropout and ReLU-based nonlinearity. Dropout for fully-connected layers is disabled by default. The final layer constitutes a scalar that models the steering angle. As this is a regression neural network, the chosen loss function is mean-squared error (MSE). The Adam optimizer is used to train the model.

The described model is defined with Keras using the following function:

```python
def nn_model(prob=0.5, dropout_for_dense=True):

    model = Sequential()

    model.add( Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 3)) )
    model.add( Cropping2D(cropping=((70, 25), (0, 0))) )

    model.add( Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu') )
    model.add(Dropout(prob))
    model.add( Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu') )
    model.add(Dropout(prob))
    model.add( Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu') )
    model.add(Dropout(prob))
    model.add( Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu') )
    model.add(Dropout(prob))

    model.add( Flatten() )

    model.add( Dense(100, activation='relu') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(50, activation='relu') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(10, activation='relu') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(1) )

    model.compile(loss='mse', optimizer='adam')

    return model
```

Detailed dimensionality of the network's layers is outlined below (a call to `model.summary()`):


```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_2 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_2 (Cropping2D)        (None, 65, 320, 3)    0           lambda_2[0][0]                   
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_2[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 158, 24)   0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 14, 77, 36)    21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 14, 77, 36)    0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 5, 37, 48)     43248       dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 5, 37, 48)     0           convolution2d_7[0][0]            
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 2, 18, 64)     27712       dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 2, 18, 64)     0           convolution2d_8[0][0]            
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 2304)          0           dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 100)           230500      flatten_2[0][0]                  
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 100)           0           dense_5[0][0]                    
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 50)            5050        dropout_5[0][0]                  
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 50)            0           dense_6[0][0]                    
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 10)            510         dropout_6[0][0]                  
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 10)            0           dense_7[0][0]                    
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 1)             11          dropout_7[0][0]                  
====================================================================================================
Total params: 330,491
Trainable params: 330,491
Non-trainable params: 0
```

## Training strategy

The model was trained on a GPU-enabled AWS instance (type `g2.2xlarge`). Many experiments with different data sets and training strategies were evaluated, mostly yielding unsatisfactory results. As noted earlier, the data used for training played the most significant role. By tweaking steering angles of left and right images the performance became better, but the results varied from one training attempt to another. Most likely, depending og how data was initially shuffled and split to the training and validation set, the final output were either better or worse.

With the successful training attempt (saved in `successful_attempt`), the third and fourth epochs resulted in rather good driving behavior, while with the latter epochs the quality degraded.
