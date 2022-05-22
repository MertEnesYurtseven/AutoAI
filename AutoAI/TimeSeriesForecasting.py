import autokeras as ak
import numpy as np
import tensorflow as tf

def SingleVariableTimeSeriesForecaster(x_train,x_test,y_train,y_test,predict_from,predict_until,lookback):
    clf = ak.TimeseriesForecaster(lookback=lookback,predict_from=predict_from,predict_until=predict_until,max_trials=1,objective="val_loss")
    # Train the TimeSeriesForecaster with train data
    clf.fit(x=x_train,y=y_train,validation_data=(x_test, y_test),batch_size=32,epochs=10)
    model = clf.export_model()
    try:
        model.save("output/SingleVariableTimeSeriesForecaster", save_format="tf")
        print("model saved as auto keras use this format")
    except Exception:
        model.save("output/SingleVariableTimeSeriesForecaster.h5")
        print("model saved as h5 use this format")
    return clf.evaluate(x_test,y_test)

def MultiVariateTimeSeriesForecasting(train,test,n_past,n_future,n_features):
    def split_series(series, n_past, n_future):
        #
        # n_past ==> no of past observations
        #
        # n_future ==> no of future observations
        #
        X, y = list(), list()
        for window_start in range(len(series)):
            past_end = window_start + n_past
            future_end = past_end + n_future
            if future_end > len(series):
                break
            # slicing the past and future parts of the window
            past, future = series[window_start:past_end, :], series[past_end:future_end, :]
            X.append(past)
            y.append(future)
        return np.array(X), np.array(y)

    X_train, y_train = split_series(train.values, n_past, n_future)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
    X_test, y_test = split_series(test.values, n_past, n_future)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
    # E2D2
    # n_features ==> no of features at each timestep in the data.
    #
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    encoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]
    #
    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
    #
    decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
    #
    model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
    #
    model_e2d2.summary()
    model_e2d2.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
    history_e2d2 = model_e2d2.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=32,verbose=0, callbacks=[reduce_lr])
    model_e2d2.save("output/MultiVariateTimeSeriesForecasting.h5")
