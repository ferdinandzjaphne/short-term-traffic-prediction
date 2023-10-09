import pandas as pd
from main import URBAN_MIX_CSV, URBAN_CORE_CSV
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.metrics import Precision, Recall


def show_image_representation(file_name): 
    df = pd.read_csv(file_name, header=None) 
    df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)

    num_train_rows = 6048 # 21 days
    # Select the rows for the training independent variable set
    X_train = df.iloc[:, 0:num_train_rows].reset_index(drop=True)

    # Select the rows for the test independent variable set (remaining rows)
    X_test = df.iloc[:, num_train_rows:].reset_index(drop=True)

    img_train = Image.fromarray(np.uint8(X_train.to_numpy()))
    img_test = Image.fromarray(np.uint8(X_test.to_numpy()))

    plt.xlabel('time')
    plt.ylabel('space')
    plt.imshow(img_train)
    plt.imshow(img_test)
    plt.show()

def load_dataset():
    output_folder_X_train= 'output_folder_X_train'
    output_folder_Y_train = 'output_folder_Y_train'
    output_folder_X_val = 'output_folder_X_val'
    output_folder_Y_val = 'output_folder_Y_val'
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []

    files = os.listdir(output_folder_X_train)
    for filename in files:
        # Passing the entire path of the image file
        file = os.path.join(output_folder_X_train, filename)
        
        # Load original via OpenCV, so we can draw on it and display it on our screen
        original = cv2.imread(file)
        X_train.append(original)


    files = os.listdir(output_folder_Y_train)
    for filename in files:
        # Passing the entire path of the image file
        file = os.path.join(output_folder_Y_train, filename)
        
        # Load original via OpenCV, so we can draw on it and display it on our screen
        original = cv2.imread(file)
        
        original = original/255

        Y_train.append(original)

    files = os.listdir(output_folder_X_val)
    for filename in files:
        # Passing the entire path of the image file
        file = os.path.join(output_folder_X_val, filename)
        
        # Load original via OpenCV, so we can draw on it and display it on our screen
        original = cv2.imread(file)

        original = original/255

        X_val.append(original)

    files = os.listdir(output_folder_Y_val)
    for filename in files:
        # Passing the entire path of the image file
        file = os.path.join(output_folder_Y_val, filename)
        
        # Load original via OpenCV, so we can draw on it and display it on our screen
        original = cv2.imread(file)

        original = original/255

        Y_val.append(original)
    
    return np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val)

def train_cnn(file_name):
    X_train, Y_train, X_val, Y_val = load_dataset()

    input_shape=(304,180,3)

    model = Sequential()
   
    # Layer 1: Convolutional Layer with 256 filters and ReLU activation
    model.add(Conv2D(256, (3, 3), activation='relu', input_shape= input_shape))

    # Layer 2: MaxPooling Layer
    model.add(MaxPooling2D((2, 2)))

    # Layer 3: Convolutional Layer with 128 filters and ReLU activation
    model.add(Conv2D(128, (3, 3), activation='relu'))

    # Layer 4: MaxPooling Layer
    model.add(MaxPooling2D((2, 2)))

    # Layer 5: Convolutional Layer with 64 filters and ReLU activation
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Layer 4: Flatten
    model.add(Flatten())

    # Layer 6: Fully Connected Layer with output shape (304*2)
    model.add(Dense(304 * 12 * 3))

    # Reshape the output to the desired shape (304, 2)
    model.add(Reshape((304, 12, 3)))

    model.summary()

    metrics=['accuracy',
               	Precision(name='precision'),
               	Recall(name='recall')]

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    history = model.fit(X_train, Y_train, epochs=50, validation_data =(X_val, Y_val))

    show_training_result(history)
    

def show_training_result(history):
    train_perf = history.history[str('accuracy')]
    validation_perf = history.history['val_accuracy']
    intersection_idx = np.argwhere(np.isclose(train_perf,
                                                validation_perf, atol=1e-2)).flatten()[0]
    intersection_value = train_perf[intersection_idx]

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)  # 2 rows, 2 columns, subplot 1
    ax1.set_title('Accuracy')
    ax1.plot(train_perf, label='accuracy')
    ax1.plot(validation_perf, label = 'val_'+str('accuracy'))
    ax1.axvline(x=intersection_idx, color='r', linestyle='--', label='Intersection')

    ax1.annotate(f'Optimal Value: {intersection_value:.4f}',
            xy=(intersection_idx, intersection_value),
            xycoords='data',
            fontsize=10,
            color='green')
                    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(loc='lower right')


    # Access the MSE values from the training history
    mse_values = history.history['loss']  # Change 'loss' to the appropriate metric name if needed
    val_mse_values = history.history['val_loss']  # If you have validation data

    # Plot the MSE values
    epochs = range(1, len(mse_values) + 1)

    ax2 = fig.add_subplot(2, 2, 2)  # 2 rows, 2 columns, subplot 2
    ax2.plot(epochs, mse_values, 'b', label='Training MSE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MSE')
    ax2.set_title('Training Mean Squared Error')
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)  # 2 rows, 2 columns, subplot 3
    # Optionally, plot validation MSE values if available
    if 'val_loss' in history.history:
        ax3.plot(epochs, val_mse_values, 'r', label='Validation MSE')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('MSE')
        ax3.set_title('Validation Mean Squared Error')
        ax3.legend()
    fig.tight_layout()

    plt.show()

def generate_image_dataset(file_name):
    df = pd.read_csv(file_name, header=None) 
    df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)

    num_train_rows = 6048 # 21 days
    num_val_rows = 576 # 2 days
    num_test_rows = 2016 # 7 days

    num_X = 180 # 1 day
    num_Y = 12

    train_dataset = df.iloc[:, 0:num_train_rows].reset_index(drop=True)
    validate_dataset = df.iloc[:, num_train_rows:num_train_rows+num_val_rows].reset_index(drop=True)
    test_dataset = df.iloc[:, num_train_rows+num_val_rows:num_train_rows+num_val_rows+num_test_rows].reset_index(drop=True)

    # Define the number of columns per row
    columns_per_row = 192 # 8 time steps, 40 minutes

    output_folder_X_train= 'output_folder_X_train'
    output_folder_Y_train = 'output_folder_Y_train'
    output_folder_X_val = 'output_folder_X_val'
    output_folder_Y_val = 'output_folder_Y_val'

    if not os.path.exists(output_folder_X_train):
        os.makedirs(output_folder_X_train)

    if not os.path.exists(output_folder_Y_train):
        os.makedirs(output_folder_Y_train)
    
    if not os.path.exists(output_folder_X_val):
        os.makedirs(output_folder_X_val)
    
    if not os.path.exists(output_folder_Y_val):
        os.makedirs(output_folder_Y_val)


    # train dataset
    for start_column in range(0, len(train_dataset.columns), columns_per_row):
        end_column_train = start_column + num_X
        end_column_test = end_column_train + num_Y
        
        X = df.iloc[:, start_column:end_column_train].to_numpy()
        Y = df.iloc[:, end_column_train :end_column_test].to_numpy()

        img_train_X = Image.fromarray(np.uint8(X))
        # Save the image(s) to the folder
        file_name = 'dataset_X_train' + str(end_column_train) + '.png'
        img_train_X.save(os.path.join(output_folder_X_train, file_name))

        img_train_Y = Image.fromarray(np.uint8(Y))
        # Save the image(s) to the folder
        file_name = 'dataset_Y_train' + str(end_column_test) + '.png'
        img_train_Y.save(os.path.join(output_folder_Y_train, file_name))
    
    for start_column in range(0, len(validate_dataset.columns), columns_per_row):
        end_column_train = start_column + num_X
        end_column_test = end_column_train + num_Y

        X = df.iloc[:, start_column:end_column_train].to_numpy()
        Y = df.iloc[:, end_column_train:end_column_test].to_numpy()

        img_Val_X = Image.fromarray(np.uint8(X))
        # Save the image(s) to the folder
        file_name = 'dataset_X_val' + str(end_column_train) + '.png'
        img_Val_X.save(os.path.join(output_folder_X_val, file_name))

        img_train_Y = Image.fromarray(np.uint8(Y))
        # Save the image(s) to the folder
        file_name = 'dataset_Y_val' + str(end_column_train) + '.png'
        img_train_Y.save(os.path.join(output_folder_Y_val, file_name))


    
# show_image_representation(URBAN_CORE_CSV)
train_cnn(URBAN_CORE_CSV)
# generate_image_dataset(URBAN_CORE_CSV)