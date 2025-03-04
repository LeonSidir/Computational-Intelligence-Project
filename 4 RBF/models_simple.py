import warnings
import keras
import numpy
import sklearn.preprocessing as sklpp
from RBFLayer import RBFLayer
from matplotlib import pyplot as pp
from keras import losses
from keras import metrics
from keras import backend
from keras import layers
from keras import optimizers
from keras import datasets
from keras.datasets import boston_housing
from kmeans_initializer import InitCentersKMeans

warnings.filterwarnings("ignore")
print("Process Running...")

def R_squared(y_actual, y_predicted):
    """This function calculates R2 square based on the following formula:
        R^2 = 1- SSres / SStot

    where:
        -> SSres is the sum of squares of the residual errors
        -> SStot is the total sum of the errors
    """

    SSres = backend.sum(backend.square(y_actual - y_predicted))
    SStot = backend.sum(backend.square(y_actual - backend.mean(y_actual)))
    r2 = 1 - SSres / (SStot + backend.epsilon())  # kr.epsilon() addition ensures that we don't divide by zero
    return r2


def make_plot(ModelHistory, ModelName):
    # make diagram for R^2
    pp.title(r'$R^2$ for model: ' + ModelName)
    pp.plot(ModelHistory.history['R_squared'], color='blue')
    pp.plot(ModelHistory.history['val_R_squared'], color='green')
    pp.xlabel('Epoch')
    pp.ylabel('Coefficient of Determination')
    pp.legend(['Train', 'Validation'], loc='center right')
    pp.savefig('R2 ' + ModelName + ".png")
    print("Plot completed!")
    print("Close the plot window to continue.")
    pp.show()

    # make diagram for Loss
    pp.title('Loss for model: ' + ModelName)
    pp.plot(ModelHistory.history['loss'], color='red')
    pp.plot(ModelHistory.history['val_loss'], color='black')
    pp.xlabel('Epoch')
    pp.ylabel('Cost')
    pp.legend(['Train', 'Validation'], loc='center right')
    pp.savefig('Loss ' + ModelName + ".png")
    print("Plot completed!")
    print("Close the plot window to continue.")
    pp.show()


ModelR2 = []
ModelR2Validation = []
ModelRMSE = []
ModelRMSEValidation = []

learning_rate = 0.001
OutLayerSize = 1        #Output Layer
SizeofDenseLayer = 128  #Hidden Layer
BatchSize = 16

(trainX, trainY), (testX, testY) = datasets.boston_housing.load_data(path="boston_housing.npz", test_split=.25)

trainSize = numpy.max(trainX.shape)
layer_sizes = numpy.floor(numpy.array([trainSize * 0.1, trainSize * 0.5, trainSize * 0.9])).astype(int)

testX = sklpp.StandardScaler().fit_transform(testX)
trainX = sklpp.StandardScaler().fit_transform(trainX)

# Model 1
print("*************")
print("** Model 1 **")
print("*************")
Model1 = keras.Sequential(name='RBF-' + str(layer_sizes[0]) + '-neurons')
Model1.add(RBFLayer(layer_sizes[0], initializer=InitCentersKMeans(trainX), input_shape=(13,)))
Model1.add(layers.Dense(SizeofDenseLayer, ))
Model1.add(layers.Dense(OutLayerSize, ))
Model1.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(), metrics=[metrics.RootMeanSquaredError(), R_squared])
Model_1_History = Model1.fit(trainX, trainY, epochs=100, batch_size=BatchSize, validation_split=0.2)
make_plot(ModelHistory=Model_1_History, ModelName=Model1.name)
ModelR2.append(Model_1_History.history['R_squared'][-1])
ModelR2Validation.append(Model_1_History.history['val_R_squared'][-1])
ModelRMSE.append(Model_1_History.history['root_mean_squared_error'][-1])
ModelRMSEValidation.append(Model_1_History.history['val_root_mean_squared_error'][-1])

# Model 2
print("*************")
print("** Model 2 **")
print("*************")
Model2 = keras.Sequential(name='RBF-' + str(layer_sizes[1]) + '-neurons')
Model2.add(RBFLayer(layer_sizes[1], initializer=InitCentersKMeans(trainX), input_shape=(13,)))
Model2.add(layers.Dense(SizeofDenseLayer, ))
Model2.add(layers.Dense(OutLayerSize, ))
Model2.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(), metrics=[metrics.RootMeanSquaredError(), R_squared])
Model_2_History = Model2.fit(trainX, trainY, epochs=100, batch_size=BatchSize, validation_split=0.2)
make_plot(ModelHistory=Model_2_History, ModelName=Model2.name)
ModelR2.append(Model_2_History.history['R_squared'][-1])
ModelR2Validation.append(Model_2_History.history['val_R_squared'][-1])
ModelRMSE.append(Model_2_History.history['root_mean_squared_error'][-1])
ModelRMSEValidation.append(Model_2_History.history['val_root_mean_squared_error'][-1])

# Model 3
print("*************")
print("** Model 3 **")
print("*************")
Model3 = keras.Sequential(name='RBF-' + str(layer_sizes[2]) + '-neurons')
Model3.add(RBFLayer(layer_sizes[2], initializer=InitCentersKMeans(trainX), input_shape=(13,)))
Model3.add(layers.Dense(SizeofDenseLayer, ))
Model3.add(layers.Dense(OutLayerSize, ))
Model3.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(), metrics=[metrics.RootMeanSquaredError(), R_squared])
Model_3_History = Model3.fit(trainX, trainY, epochs=100, batch_size=BatchSize, validation_split=0.2)
make_plot(ModelHistory=Model_3_History, ModelName=Model3.name)
ModelR2.append(Model_3_History.history['R_squared'][-1])
ModelR2Validation.append(Model_3_History.history['val_R_squared'][-1])
ModelRMSE.append(Model_3_History.history['root_mean_squared_error'][-1])
ModelRMSEValidation.append(Model_3_History.history['val_root_mean_squared_error'][-1])

print('=======================================================================================')
print('=====================================RESULTS===========================================')
print('=======================================================================================')
i = 1

for R2, R2_validation, RMSE, RMSE_validation in zip(ModelR2, ModelR2Validation, ModelRMSE, ModelRMSEValidation):
    print(f"Model {i} R2 = {R2}, R2_validation = {R2_validation}, RMSE = {RMSE}, RMSE_validation = {RMSE_validation}".format(i=i, R2=R2, R2_validation=R2_validation, RMSE=RMSE, RMSE_validation=RMSE_validation))
    i = i + 1

print('=======================================================================================')
print('=======================================================================================')
print('=======================================================================================')

