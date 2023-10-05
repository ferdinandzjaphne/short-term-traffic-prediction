import pandas as pd
from main import URBAN_MIX_CSV, URBAN_CORE_CSV
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

def train_cnn(file_name):
    X_train, Y_train, X_val, Y_val = generate_image_dataset(file_name)

    # Create a Sequential model
    model = Sequential()

    # Add the first convolutional layer with 256 filters
    model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))

    # Add the second convolutional layer with 128 filters
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Add the third convolutional layer with 64 filters
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the output from the convolutional layers
    model.add(Flatten())

    # Add a fully connected layer with a dense output
    model.add(Dense(128, activation='relu'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))

    # Display the model summary
    model.summary()

def generate_image_dataset(file_name):
    df = pd.read_csv(file_name, header=None) 
    df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)

    num_train_rows = 6048 # 21 days
    num_val_rows = 576 # 2 days
    num_test_rows = 2016 # 7 days

    num_X = 6
    num_Y = 2

    train_dataset = df.iloc[:, 0:num_train_rows].reset_index(drop=True)
    validate_dataset = df.iloc[:, num_train_rows:num_train_rows+num_val_rows].reset_index(drop=True)
    test_dataset = df.iloc[:, num_train_rows+num_val_rows:num_train_rows+num_val_rows+num_test_rows].reset_index(drop=True)

    # Define the number of columns per row
    columns_per_row = 8 # 8 time steps, 40 minutes

    output_folder = 'output_images'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    X_train = []
    Y_train = []
    X_val = []
    Y_val = []

    # train dataset
    for start_column in range(0, len(train_dataset.columns), columns_per_row):
        end_column_train = start_column + num_X
        end_column_test = end_column_train + num_Y

        X = df.iloc[:, start_column:end_column_train]
        Y = df.iloc[:, end_column_train:end_column_test]

        # img_train = Image.fromarray(np.uint8(selected_columns.to_numpy()))
        # plt.imshow(img_train)
        # plt.show()
        # Save the image(s) to the folder
        # file_name = 'dataset_' + str(end_column) + '.png'
        # img_train.save(os.path.join(output_folder, file_name))

        X_train.append(X)
        Y_train.append(Y)
    
    for start_column in range(0, len(validate_dataset.columns), columns_per_row):
        end_column_train = start_column + num_X
        end_column_test = end_column_train + num_Y

        X = df.iloc[:, start_column:end_column_train]
        Y = df.iloc[:, end_column_train:end_column_test]

        # img_train = Image.fromarray(np.uint8(selected_columns.to_numpy()))
        # plt.imshow(img_train)
        # plt.show()
        # Save the image(s) to the folder
        # file_name = 'dataset_' + str(end_column) + '.png'
        # img_train.save(os.path.join(output_folder, file_name))

        X_val.append(X)
        Y_val.append(Y)

    # print(len(X_train))
    # print(X_train[0])
    # print(len(Y_train))
    # print(Y_train[0])

    return X_train, Y_train, X_val, Y_val


    
# show_image_representation(URBAN_CORE_CSV)
train_cnn(URBAN_CORE_CSV)