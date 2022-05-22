import autokeras as ak
import tensorflow as tf



def StandartImageClassifier(x_train, y_train,x_test, y_test):
    clf = ak.ImageClassifier(overwrite=True, max_trials=1)
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


def AdvancedImageClassifier(x_train, y_train,x_test, y_test):
    input_node = ak.ImageInput()
    output_node = ak.Normalization()(input_node)
    output_node = ak.ImageAugmentation(horizontal_flip=False)(output_node)
    output_node = ak.ResNetBlock(version="v2")(output_node)
    output_node = ak.ClassificationHead()(output_node)
    clf = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=1)
    clf.fit(x_train, y_train, epochs=10)
    model = clf.export_model()
    try:
        model.save("model_autokeras", save_format="tf")
        print("output/model saved as auto keras use this format")
    except:
        model.save("model_autokeras.h5")
        print("output/model saved as h5 use this format")
    return clf.evaluate(x_test, y_test)