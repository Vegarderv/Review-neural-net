import pickle
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.preprocessing import sequence

with open(file="keras-data.pickle", mode="rb") as file:
    data = pickle.load(file)
vocab_size = data["vocab_size"]
max_length = data["max_length"]

def prepare_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare the data: It will end up as shuffled and split between train and test sets.
    Data is normalized to (0, 1) and coded as floats.
    Shape of each image in the data is (28, 28, 1), since we only have one color channel.

    :return: Two tuples: First with training data (x, y), second with test (x, y)
    """
    

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    x_train = np.array(sequence.pad_sequences(x_train, max_length))
    y_train = np.array(y_train)
    x_test = np.array(sequence.pad_sequences(x_test, max_length))
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

def define_model() -> tf.keras.models.Sequential:


    # Use tf.keras and the Sequential class to define the model layer-by-layer.
    # This hides all the dirty details from us, yet gives the flexibility needed to make an OK model.
    model = tf.keras.models.Sequential()

    model.add(                          # Add a layer to the sequential model
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length)
    )

    model.add(
        tf.keras.layers.LSTM(50)
    )
    
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer="adam")

    try:
        model.load_weights("./weights.h5")
        print('Weights loaded successfully')
    except IOError:
        print("Weight loading didn't work")

    model.summary()
    return model


def learn_model_from_data(no_epochs: int = 50) -> tf.keras.models.Sequential:


    # Define model structure
    model = define_model()

    # Get hold of data
    train_data, test_data = prepare_data()


    model.fit(
        *train_data,
        batch_size=100,
        epochs=no_epochs,
        verbose=1,
        validation_data=test_data
    )

    """
    Check quality and report to screen. 
    Since I know this works well I don't care about careful examination of results here.
    In a more real situation I'd received the history-object returned from model.fit and analyzed it more carefully.  
    """
    loss, accuracy = model.evaluate(*test_data, verbose=0)
    print(f'\nTest loss: {loss:.6f}')
    print(f'Test accuracy: {accuracy * 100:.2f} %')

    """
    Save the learned weights and leave
    """
    model.save_weights('./weights.h5')
    return model


if __name__ == "__main__":
    model_ = learn_model_from_data(no_epochs=1)
