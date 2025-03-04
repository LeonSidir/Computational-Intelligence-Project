import warnings
import numpy
import keras_tuner
from tensorflow import keras
from keras import backend
from keras import losses
from keras import metrics
from keras import optimizers
from keras import layers
from keras import datasets
from keras.datasets import boston_housing
from RBFLayer import RBFLayer
from sklearn.preprocessing import normalize
from kmeans_initializer import InitCentersKMeans
from matplotlib import pyplot as pp

warnings.filterwarnings("ignore")
print("Process Running...")

OutLayerSize = 1
learning_rate = 0.001
RBF_Layer_Values = [.5, .15, .3, .5]
DropoutRateValues = [.2, .35, .5]
HiddenLayerValues = [32, 64, 128, 256]

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


def create_model(mod):
    model = keras.Sequential()
    rbf_layer_size = mod.Choice('rbf_units', values=RBF_Layer_Values)
    hidden_layer_size = mod.Choice('hidden_layer_units', values=HiddenLayerValues)
    dropout_rates = mod.Choice('dropout_rate', values=DropoutRateValues)

    clusters = int(numpy.floor(rbf_layer_size * numpy.max(trainX.shape)))

    rbf_layer = RBFLayer(clusters, initializer=InitCentersKMeans(trainX), input_shape=(13,))
    hidden_layer = layers.Dense(hidden_layer_size, )
    dropout_layer = layers.Dropout(rate=dropout_rates)
    output_layer = layers.Dense(OutLayerSize, )

    model.add(rbf_layer)
    model.add(hidden_layer)
    model.add(dropout_layer)
    model.add(output_layer)
    model.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(), metrics=[metrics.RootMeanSquaredError(), R_squared])
    return model


(trainX, trainY), (testX, testY) = datasets.boston_housing.load_data(path="boston_housing.npz", test_split=.25)

testX = normalize(testX)
trainX = normalize(trainX)

RBF_Tuner = keras_tuner.RandomSearch(create_model, objective=keras_tuner.Objective("root_mean_squared_error", direction="min"), directory='Fine Tuning RBF network', project_name='Fine Tuning RBF', overwrite=True)
RBF_Tuner.search(trainX, trainY, epochs=100, validation_split=0.2)

optimal_mod = RBF_Tuner.get_best_hyperparameters(num_trials=1)[0]
RBF_Layer_Size = optimal_mod.get('rbf_units')
HiddenLayerSize = optimal_mod.get('hidden_layer_units')
DropoutRate = optimal_mod.get('dropout_rate')

clusters = int(numpy.floor(RBF_Layer_Size * numpy.max(trainX.shape)))

RBF_Layer = RBFLayer(clusters, initializer=InitCentersKMeans(trainX), input_shape=(13,))
HiddenLayer = layers.Dense(HiddenLayerSize, )
DropoutLayer = layers.Dropout(rate=DropoutRate)
OutLayer = layers.Dense(OutLayerSize, )

model = keras.Sequential()
model.add(RBF_Layer)
model.add(HiddenLayer)
model.add(DropoutLayer)
model.add(OutLayer)

model.compile(optimizer=optimizers.SGD(learning_rate=learning_rate), loss=losses.MeanSquaredError(), metrics=[metrics.RootMeanSquaredError(), R_squared])

history = model.fit(trainX, trainY, epochs=100, batch_size=128, validation_split=0.2)
make_plot(ModelHistory=history, ModelName="Fine Tuning RBF network")

loss, root_mean_squared_error, r_squared = model.evaluate(testX, testY, verbose=0)
print('=======================================================================================')
print('=====================================RESULTS===========================================')
print('=======================================================================================')
print('Coefficient of Determination(R^2): ', r_squared)
print('RMSE: ', root_mean_squared_error)
print('Loss: ', loss)
print('---------------------------------------------------------------------------------------')
print("Hyperparamters:")
print(f"Best RBF Layer size = {RBF_Layer_Size} or {RBF_Layer_Size*100}% of size of trainX")
print(f"Best Hidden Layer units = {HiddenLayerSize}")
print(f"Best Dropout rate = {DropoutRate}")
print('=======================================================================================')
print('=======================================================================================')
print('=======================================================================================')

