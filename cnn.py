import pandas as pd
from main import URBAN_MIX_CSV, URBAN_CORE_CSV
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape,GlobalAveragePooling2D
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_percentage_error


num_train_rows = 6048 # 21 days
num_val_rows = 576 # 2 days
num_test_rows = 2016 # 7 days

num_X = 20 
num_Y = 3

road_segment = 1007

# Define the number of columns per row
columns_per_row = num_X + num_Y


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

def load_dataset(file_name):
    output_folder_X_train= 'output_folder_X_train_'  + file_name
    output_folder_Y_train = 'output_folder_Y_train_' + file_name
    output_folder_X_val = 'output_folder_X_val_' + file_name
    output_folder_Y_val = 'output_folder_Y_val_' + file_name
    output_folder_X_test = 'output_folder_X_test_' + file_name
    output_folder_Y_test = 'output_folder_Y_test_' + file_name
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    X_test = []
    Y_test = []

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
        Y_train.append(original)

    files = os.listdir(output_folder_X_val)
    for filename in files:
        # Passing the entire path of the image file
        file = os.path.join(output_folder_X_val, filename)
        
        # Load original via OpenCV, so we can draw on it and display it on our screen
        original = cv2.imread(file)
        X_val.append(original)

    files = os.listdir(output_folder_Y_val)
    for filename in files:
        # Passing the entire path of the image file
        file = os.path.join(output_folder_Y_val, filename)
        
        # Load original via OpenCV, so we can draw on it and display it on our screen
        original = cv2.imread(file)
        Y_val.append(original)

    files = os.listdir(output_folder_X_test)
    for filename in files:
        # Passing the entire path of the image file
        file = os.path.join(output_folder_X_test, filename)
        
        # Load original via OpenCV, so we can draw on it and display it on our screen
        original = cv2.imread(file)
        X_test.append(original)

    files = os.listdir(output_folder_Y_test)
    for filename in files:
        # Passing the entire path of the image file
        file = os.path.join(output_folder_Y_test, filename)
        
        # Load original via OpenCV, so we can draw on it and display it on our screen
        original = cv2.imread(file)
        Y_test.append(original)
    
    return np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val), np.array(X_test), np.array(Y_test)

def train_cnn(file_name):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset(file_name)

    input_shape=(road_segment,num_X,3)

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

    # Layer 6: MaxPooling Layer
    # model.add(MaxPooling2D((2, 2)))

    # Layer 7: Flatten
    model.add(Flatten())

    # Layer 8: Fully Connected Layer with output shape (304*2)
    model.add(Dense(road_segment * num_Y * 3))

    model.add(Reshape((road_segment, num_Y, 3)))

    model.summary()

    metrics=['accuracy',
               	Precision(name='precision'),
               	Recall(name='recall')]

    opt = Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss='mean_squared_error', metrics=metrics)

    checkpoint_callback = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    history = model.fit(X_train, Y_train, epochs=80, validation_data =(X_val, Y_val), callbacks=[checkpoint_callback])
    best_val_loss = min(history.history['val_loss'])
    print('best val loss: ', best_val_loss)

    model.save('best_model.h5')

    # Load the best model (optional)
    best_model = load_model("best_model.h5")

    test_loss, test_accuracy, precision_score, recall_score = best_model.evaluate(X_test, Y_test)
    print("15 mins result")
    print(test_loss)
    print(test_accuracy)
    print(precision_score)
    print(recall_score)

    show_training_result(history)

def check_model_on_task(file_name): 
    _, _, _, _, X_test, Y_test = load_dataset(file_name)
    X_test_30_mins = Y_test[:, :, :6, :]
    Y_test_30_mins = Y_test[:, :, :6, :]
    X_test_15_mins = Y_test[:, :, :3, :]
    Y_test_15_mins = Y_test[:, :, :3, :]
    # Load the best model (optional)
    best_model = load_model("best_model.h5")

    test_loss, test_accuracy, precision_score, recall_score = best_model.evaluate(X_test, Y_test)
    print("60 mins result")
    print(test_loss)
    print(test_accuracy)
    print(precision_score)
    print(recall_score)

    # change to 30 minutes
    # Define a custom layer to reshape the output
    # reshape_layer = Reshape((304, 6, 3))(best_model.output)

    # Create a new model with the modified output shape
    # modified_model = Model(inputs=best_model.input, outputs=reshape_layer)

    # # Optionally, compile the model for training
    # metrics=['accuracy',
    #                 Precision(name='precision'),
    #                 Recall(name='recall')]

    # opt = Adam(learning_rate=0.001)
    # modified_model.compile(optimizer=opt, loss='mean_squared_error', metrics=metrics)

    # Make predictions on the test data
    test_loss, test_accuracy, precision_score, recall_score = best_model.evaluate(X_test_30_mins, Y_test_30_mins)
    print("30 mins result")
    print(test_loss)
    print(test_accuracy)
    print(precision_score)
    print(recall_score)

     # change to 15 minutes
    best_model.add(Dense(304 * 3 * 3), name='dense_15_mins')

    metrics=['accuracy',
                    Precision(name='precision'),
                    Recall(name='recall')]

    opt = Adam(learning_rate=0.001)
    best_model.compile(optimizer=opt, loss='mean_squared_error', metrics=metrics)

    # Make predictions on the test data
    y_pred_2 = best_model.predict(X_test)
    y_true_reshaped_2 = Y_test_15_mins.reshape(Y_test_15_mins.shape[0], -1)  # Flattening along axis 1

    y_pred_reshaped_2 = y_pred_2.reshape(y_pred_2.shape[0], -1)  # Flattening along axis 1

    # Calculate MSE
    mse = mean_squared_error(y_true_reshaped_2, y_pred_reshaped_2)

    # Print the MSE
    print(f"Mean Squared Error (MSE) 15 mins prediction: {mse}")

def show_images_in_folder(folder_path):
    # Get a list of files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_files = image_files[:9]

    num_images = len(image_files)
    num_cols = 3  # Number of columns in the plot
    num_rows = (num_images // num_cols) + 1

    plt.figure(figsize=(10, 8))

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)

        # Open the image using PIL
        img = Image.open(image_path)

        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.title(image_file)
        plt.axis('off')  # Hide axis

    plt.tight_layout()
    plt.show()




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


    train_dataset = df.iloc[:, 0:num_train_rows].reset_index(drop=True)
    validate_dataset = df.iloc[:, num_train_rows:num_train_rows+num_val_rows].reset_index(drop=True)
    test_dataset = df.iloc[:, num_train_rows+num_val_rows:num_train_rows+num_val_rows+num_test_rows].reset_index(drop=True)


    output_folder_X_train= 'output_folder_X_train_' + file_name
    output_folder_Y_train = 'output_folder_Y_train_' + file_name
    output_folder_X_val = 'output_folder_X_val_' + file_name
    output_folder_Y_val = 'output_folder_Y_val_' + file_name
    output_folder_X_test = 'output_folder_X_test_' + file_name
    output_folder_Y_test = 'output_folder_Y_test_' + file_name

    if not os.path.exists(output_folder_X_train):
        os.makedirs(output_folder_X_train)

    if not os.path.exists(output_folder_Y_train):
        os.makedirs(output_folder_Y_train)
    
    if not os.path.exists(output_folder_X_val):
        os.makedirs(output_folder_X_val)
    
    if not os.path.exists(output_folder_Y_val):
        os.makedirs(output_folder_Y_val)

    if not os.path.exists(output_folder_X_test):
        os.makedirs(output_folder_X_test)

    if not os.path.exists(output_folder_Y_test):
        os.makedirs(output_folder_Y_test)

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
    
    # val dataset
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

    # test dataset
    for start_column in range(0, len(test_dataset.columns), columns_per_row):
        end_column_train = start_column + num_X
        end_column_test = end_column_train + num_Y

        X = df.iloc[:, start_column:end_column_train].to_numpy()
        Y = df.iloc[:, end_column_train:end_column_test].to_numpy()

        img_Val_X = Image.fromarray(np.uint8(X))
        # Save the image(s) to the folder
        file_name = 'dataset_X_test' + str(end_column_train) + '.png'
        img_Val_X.save(os.path.join(output_folder_X_test, file_name))

        img_train_Y = Image.fromarray(np.uint8(Y))
        # Save the image(s) to the folder
        file_name = 'dataset_Y_test' + str(end_column_train) + '.png'
        img_train_Y.save(os.path.join(output_folder_Y_test, file_name))
if __name__ == "__main__":
    # show_image_representation(URBAN_CORE_CSV)
    # generate_image_dataset(URBAN_CORE_CSV)
    # train_cnn(URBAN_MIX_CSV)
    # check_model_on_task(URBAN_CORE_CSV)

    # Replace 'folder_path' with the path to your image folder
    folder_path = 'output_folder_Y_test_urban-core.csv'
    show_images_in_folder(folder_path)