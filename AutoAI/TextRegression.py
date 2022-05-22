import autokeras as ak
import tensorflow as tf



def StandartTextRegressor(x_train, y_train,x_test, y_test):
    clf = ak.TextRegressor(overwrite=True, max_trials=1)
    clf.fit(x_train, y_train, epochs=10)
    predicted_y = clf.predict(x_test)
    model = clf.export_model()
    try:
        model.save("output/model_autokeras", save_format="tf")
        print("model saved as auto keras use this format")
    except Exception:
        model.save("output/model_autokeras.h5")
        print("model saved as h5 use this format")
    return clf.evaluate(x_test,y_test)


def AdvancedTextRegressor(x_train, y_train,x_test, y_test):
    input_node = ak.TextInput()
    output_node = ak.TextToIntSequence()(input_node)
    output_node = ak.Embedding()(output_node)
    # Use separable Conv layers in Keras.
    output_node = ak.ConvBlock(separable=True)(output_node)
    output_node = ak.RegressionHeadHead()(output_node)
    clf = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=1)
    clf.fit(x_train, y_train, epochs=2)
    model = clf.export_model()
    try:
        model.save("output/model_autokeras", save_format="tf")
        print("model saved as auto keras use this format")
    except:
        model.save("output/model_autokeras.h5")
        print("model saved as h5 use this format")
    return clf.evaluate(x_test, y_test)