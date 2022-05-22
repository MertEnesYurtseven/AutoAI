import autokeras as ak
import tensorflow as tf



def StandartStructuredRegressor(TrainFilePath,TestFilePath,yLabel):
    clf = ak.StructuredDataRegressor(overwrite=True, max_trials=1)
    clf.fit(TrainFilePath,yLabel,epochs=10,)
    predicted_y = clf.predict(TestFilePath)
    model = clf.export_model()
    try:
        model.save("output/models/model_autokeras", save_format="tf")
        print("model saved as auto keras use this format")
    except Exception:
        model.save("output/models/model_autokeras.h5")
        print("model saved as h5 use this format")
    return clf.evaluate(TestFilePath,yLabel)


def AdvancedStructuredRegressor(x_train,x_test,y_train,y_test,categorical_encoding=False):
    input_node = ak.StructuredDataInput()
    output_node = ak.StructuredDataBlock(categorical_encoding=categorical_encoding)(input_node)
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead()(output_node)
    clf = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=1)
    clf.fit(x_train, y_train, epochs=1)
    clf.predict(x_train)
    model = clf.export_model()
    try:
        model.save("output/models/model_autokeras", save_format="tf")
        print("model saved as auto keras use this format")
    except:
        model.save("output/models/model_autokeras.h5")
        print("model saved as h5 use this format")
    return clf.evaluate(x_test, y_test)