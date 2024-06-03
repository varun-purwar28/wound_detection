import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.utils import plot_model, CustomObjectScope
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Concatenate, UpSampling2D
import cv2  # If OpenCV is being used for image processing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Unet2D:

    def __init__(self, n_filters, input_dim_x, input_dim_y, num_channels):
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.n_filters = n_filters
        self.num_channels = num_channels

    def get_unet_model_5_levels(self):
        unet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))
        
        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(unet_input)
        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        
        conv5 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)
        
        up6 = Conv2D(self.n_filters*16, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
        concat6 = Concatenate()([drop4, up6])
        conv6 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        
        up7 = Conv2D(self.n_filters*8, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        concat7 = Concatenate()([conv3, up7])
        conv7 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        
        up8 = Conv2D(self.n_filters*4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        concat8 = Concatenate()([conv2, up8])
        conv8 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(concat8)
        conv8 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)
        
        up9 = Conv2D(self.n_filters*2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        concat9 = Concatenate()([conv1, up9])
        conv9 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(concat9)
        conv9 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(3, kernel_size=1, activation='sigmoid', padding='same')(conv9)
        
        return Model(outputs=conv10,  inputs=unet_input), 'unet_model_5_levels'


    def get_unet_model_4_levels(self):
        unet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))
                
        conv1 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(unet_input)
        conv1 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
        
        conv4 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.n_filters*16, kernel_size=3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        
        up5 = Conv2D(self.n_filters*16, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
        concat5 = Concatenate()([drop3, up5])
        conv5 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(concat5)
        conv5 = Conv2D(self.n_filters*8, kernel_size=3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        
        up6 = Conv2D(self.n_filters*8, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        concat6 = Concatenate()([conv2, up6])
        conv6 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(self.n_filters*4, kernel_size=3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        
        up7 = Conv2D(self.n_filters*4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        concat7 = Concatenate()([conv1, up7])
        conv7 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(self.n_filters*2, kernel_size=3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)

        conv9 = Conv2D(3, kernel_size=1, activation='sigmoid', padding='same')(conv7)
        
        return Model(outputs=conv9,  inputs=unet_input), 'unet_model_4_levels'


    def get_unet_model_yuanqing(self):
        # Model inspired by https://github.com/yuanqing811/ISIC2018
        unet_input = Input(shape=(self.input_dim_x, self.input_dim_y, self.num_channels))

        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(unet_input)
        conv1 = Conv2D(self.n_filters, kernel_size=3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv3)
        conv3 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv4)
        conv4 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv5)
        conv5 = Conv2D(self.n_filters * 8, kernel_size=3, activation='relu', padding='same')(conv5)

        up6 = Conv2D(self.n_filters * 4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        feature4 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv4)
        concat6 = Concatenate()([feature4, up6])
        conv6 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(concat6)
        conv6 = Conv2D(self.n_filters * 4, kernel_size=3, activation='relu', padding='same')(conv6)

        up7 = Conv2D(self.n_filters * 2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        feature3 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(conv3)
        concat7 = Concatenate()([feature3, up7])
        conv7 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(concat7)
        conv7 = Conv2D(self.n_filters * 2, kernel_size=3, activation='relu', padding='same')(conv7)

        up8 = Conv2D(self.n_filters * 1, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        feature2 = Conv2D(self.n_filters * 1, kernel_size=3, activation='relu', padding='same')(conv2)
        concat8 = Concatenate()([feature2, up8])
        conv8 = Conv2D(self.n_filters * 1, kernel_size=3, activation='relu', padding='same')(concat8)
        conv8 = Conv2D(self.n_filters * 1, kernel_size=3, activation='relu', padding='same')(conv8)

        up9 = Conv2D(int(self.n_filters / 2), 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        feature1 = Conv2D(int(self.n_filters / 2), kernel_size=3, activation='relu', padding='same')(conv1)
        concat9 = Concatenate()([feature1, up9])
        conv9 = Conv2D(int(self.n_filters / 2), kernel_size=3, activation='relu', padding='same')(concat9)
        conv9 = Conv2D(int(self.n_filters / 2), kernel_size=3, activation='relu', padding='same')(conv9)
        conv9 = Conv2D(3, kernel_size=3, activation='relu', padding='same')(conv9)
        conv10 = Conv2D(1, kernel_size=1, activation='sigmoid')(conv9)

        return Model(outputs=conv10, inputs=unet_input), 'unet_model_yuanqing'
    
    
import os


# Import custom metrics and losses
from utils.learning import dice_coef, precision, recall
from utils.losses import dice_coef_loss
from utils.data import DataGen, save_results, save_history, load_data


input_dim_x = 512
input_dim_y = 512
n_filters = 32
dataset = "D:/segmentation_training_data/segmentation_training_data"
data_gen = DataGen(dataset + '/', split_ratio=0.2, x=input_dim_x, y=input_dim_y)

# Get the deep learning models

unet2d = Unet2D(n_filters=n_filters, input_dim_x=None, input_dim_y=None, num_channels=3)
model, model_name = unet2d.get_unet_model_yuanqing()

######### MobilenetV2 ##########
# model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)
# model_name = 'MobilenetV2'
# with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D, 'BilinearUpsampling': BilinearUpsampling}):
#     model = load_model('azh_wound_care_center_diabetic_foot_training_history/2020-02-10 02:57:27.555495.hdf5',
#                        custom_objects={'dice_coef': dice_coef, 'precision': precision, 'recall': recall})

######### SegNet ##########
# segnet = SegNet(n_filters, input_dim_x, input_dim_y, num_channels=3)
# model, model_name = segnet.get_SegNet()

# Plot model (optional)
# plot_model(model, to_file=model_name + '.png')

smooth = 1e-5

import tensorflow as tf

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def precision(truth, prediction):
    TP = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(truth * prediction, 0, 1)))
    total_predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(prediction, 0, 1)))
    return TP / (total_predicted_positives + tf.keras.backend.epsilon())


# Training parameters
batch_size = 2
epochs = 2000
learning_rate = 1e-4
loss = "mean_squared_error"

# Define early stopping callback
es = EarlyStopping(monitor='val_dice_coef', patience=200, mode='max', restore_best_weights=True)

# Compile and train the model
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss, metrics=[dice_coef, precision, recall])
import numpy as np

# Initialize empty lists to store training metrics
import tensorflow as tf
import numpy as np

# Define your optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Initialize empty lists to store training metrics
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

# Loop through epochs
for epoch in range(epochs):
    print("Epoch", epoch+1)
    
    # Initialize epoch-level metrics
    epoch_train_loss = []
    epoch_train_accuracy = []
    epoch_val_loss = []
    epoch_val_accuracy = []
    
    # Training loop
    for batch_x, batch_y in data_gen.generate_data(batch_size=batch_size, train=True):
        with tf.GradientTape() as tape:
            predictions = model(batch_x)
            loss = loss_fn(batch_y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Compute batch-level metrics
        batch_loss = loss.numpy()
        batch_accuracy = np.mean(tf.keras.metrics.binary_accuracy(batch_y, predictions))
        
        epoch_train_loss.append(batch_loss)
        epoch_train_accuracy.append(batch_accuracy)
    
    # Validation loop
    for batch_x_val, batch_y_val in data_gen.generate_data(batch_size=batch_size, val=True):
        val_predictions = model(batch_x_val)
        val_loss_value = loss_fn(batch_y_val, val_predictions).numpy()
        val_accuracy_value = np.mean(tf.keras.metrics.binary_accuracy(batch_y_val, val_predictions))
        
        epoch_val_loss.append(val_loss_value)
        epoch_val_accuracy.append(val_accuracy_value)
    
    # Compute epoch-level metrics
    avg_train_loss = np.mean(epoch_train_loss)
    avg_train_accuracy = np.mean(epoch_train_accuracy)
    avg_val_loss = np.mean(epoch_val_loss)
    avg_val_accuracy = np.mean(epoch_val_accuracy)
    
    train_loss.append(avg_train_loss)
    train_accuracy.append(avg_train_accuracy)
    val_loss.append(avg_val_loss)
    val_accuracy.append(avg_val_accuracy)

# Create a dictionary to store the training history
training_history = {
    'train_loss': train_loss,
    'train_accuracy': train_accuracy,
    'val_loss': val_loss,
    'val_accuracy': val_accuracy
}



# Save the model weight file and its training history
save_history(model, model_name, dataset,training_history, n_filters, epochs, loss,learning_rate, color_space='RGB', path='D:/')