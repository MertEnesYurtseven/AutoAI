# Define timesteps and the number of features
from tensorflow import keras
from keras.layers import *
from keras import Model
from keras.regularizers import *
import numpy as np
n_timesteps = 8

n_features = 7

# RNN + SLP Model

# Define input layer

def MultiInputCategoricalNumericalSupportedTimeSeriesClassifier(n_timesteps,n_features,n_outputs,X_train, X_test, y_train, y_test):
    recurrent_input = Input(shape=(n_timesteps, n_features), name="TIMESERIES_INPUT")

    static_input = Input(shape=(X_train[1].shape[1],), name="STATIC_INPUT")

    # RNN Layers

    # layer - 1

    rec_layer_one = Bidirectional(
        LSTM(128, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True),
        name="BIDIRECTIONAL_LAYER_1")(recurrent_input)

    rec_layer_one = Dropout(0.1, name="DROPOUT_LAYER_1")(rec_layer_one)

    # layer - 2

    rec_layer_two = Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
                                  name="BIDIRECTIONAL_LAYER_2")(rec_layer_one)

    rec_layer_two = Dropout(0.1, name="DROPOUT_LAYER_2")(rec_layer_two)

    # SLP Layers

    static_layer_one = Dense(64, kernel_regularizer=l2(0.001), activation='relu', name="DENSE_LAYER_1")(static_input)

    # Combine layers - RNN + SLP

    combined = Concatenate(axis=1, name="CONCATENATED_TIMESERIES_STATIC")([rec_layer_two, static_layer_one])

    combined_dense_two = Dense(64, activation='relu', name="DENSE_LAYER_2")(combined)

    output = Dense(n_outputs, activation='sigmoid', name="OUTPUT_LAYER")(combined_dense_two)

    # Compile Model

    model = Model(inputs=[recurrent_input, static_input], outputs=[output])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit([np.asarray(X_train[0]).astype('float32'), np.asarray(X_train[1]).astype('float32')],y_train, epochs=10, batch_size=16, verbose=0, validation_data=([np.asarray(X_test[0]).astype('float32'), np.asarray(X_test[1]).astype('float32')], y_test))
    model.save("output/MultiInputCategoricalNumericalSupportedTimeSeriesClassifier.h5")



def NumericalSupportedTimeSeriesClassifier(n_timesteps, n_features,n_outputs,X_train, X_test, y_train, y_test):
    recurrent_input = Input(shape=(n_timesteps, n_features), name="TIMESERIES_INPUT")
    rec_layer_one = Bidirectional(
        LSTM(128, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True),
        name="BIDIRECTIONAL_LAYER_1")(recurrent_input)

    rec_layer_one = Dropout(0.1, name="DROPOUT_LAYER_1")(rec_layer_one)

    # layer - 2

    rec_layer_two = Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
                                  name="BIDIRECTIONAL_LAYER_2")(rec_layer_one)

    rec_layer_two = Dropout(0.1, name="DROPOUT_LAYER_2")(rec_layer_two)
    combined_dense_two = Dense(64, activation='relu', name="DENSE_LAYER_2")(rec_layer_two)

    output = Dense(n_outputs, activation='sigmoid', name="OUTPUT_LAYER")(combined_dense_two)

    # Compile Model

    model = Model(inputs=[recurrent_input], outputs=[output])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(np.asarray(X_train).astype('float32'), y_train,epochs=10, batch_size=16, verbose=0, validation_data=(np.asarray(X_test).astype('float32'), y_test))
    model.save("output/NumericalSupportedTimeSeriesClassifier.h5")



