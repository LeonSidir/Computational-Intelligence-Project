import numpy
import warnings
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as pp
from keras.backend import clear_session
from keras import regularizers
from keras import layers
from keras.datasets import mnist
from time import time

warnings.filterwarnings("ignore")
print("Process Running...")

TotalClasses = 10
seed = 42
Layer_1_Size = 128
Layer_2_Size = 256
L1_a = 0.01
L2_a = [0.1, 0.01, 0.001]
L1_Mean = 10/255
DropoutRate = 0.3
Epochs = 100
ValidationSplit = 0.2
InputSize = 784  # image shape: 28 * 28 pixels


def make_plot(ModelHistory, ModelName, BatchSize):
    # Create diagram for accuracy
    pp.plot(ModelHistory.history['accuracy'],  color='blue')
    pp.plot(ModelHistory.history['val_accuracy'],  color='green')
    pp.title("Accuracy for " + ModelName + " with Batch Size = " + str(BatchSize))
    pp.legend(['Train', 'Validation'], loc='center right')
    pp.xlabel('Epoch')
    pp.ylabel('Accuracy')
    pp.savefig('Accuracy ' + ModelName + " BatchSize=" + str(BatchSize) + ".png")
    print("Plot completed!")
    print("Close the plot window to continue.")
    pp.show()

    # Create diagram for loss
    pp.plot(ModelHistory.history['loss'],  color='red')
    pp.plot(ModelHistory.history['val_loss'], color='black')
    pp.title("Loss for " + ModelName + " with Batch Size = " + str(BatchSize))
    pp.legend(['Train', 'Validation'], loc='center right')
    pp.xlabel('Epoch')
    pp.ylabel('Loss')
    pp.savefig('Loss ' + ModelName + "BatchSize=" + str(BatchSize) + ".png")
    print("Plot completed!")
    print("Close the plot window to continue.")
    pp.show()


def models_compile():
    #Compile all the different models
    SGD_LearningRate = 0.01
    RMSprop_LearningRate = 0.001
    r = [0.01, 0.99]

    Standard_Model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    RMSProp1_Model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=RMSprop_LearningRate, rho=r[0]), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    RMSProp2_Model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=RMSprop_LearningRate, rho=r[1]), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    SGD_Standard_Model.compile(optimizer=tf.optimizers.SGD(learning_rate=SGD_LearningRate), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    SGD_L1_Model.compile(optimizer=tf.optimizers.SGD(learning_rate=SGD_LearningRate), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    SGD_L2_Set1_Model.compile(optimizer=tf.optimizers.SGD(learning_rate=SGD_LearningRate), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    SGD_L2_Set2_Model.compile(optimizer=tf.optimizers.SGD(learning_rate=SGD_LearningRate), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    SGD_L2_Set3_Model.compile(optimizer=tf.optimizers.SGD(learning_rate=SGD_LearningRate), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])


# Create Layers
Standard_Layers = [
    layers.Dense(Layer_1_Size, activation=tf.nn.relu, name="Layer1"),
    layers.Dense(Layer_2_Size, activation=tf.nn.relu, name="Layer2"),
    layers.Dense(TotalClasses, activation=tf.nn.softmax, name="OutputLayer"), ]

L1_Layers = [
    layers.Dense(Layer_1_Size, activation=tf.nn.relu, kernel_regularizer=regularizers.L1(l1=L1_a), name="Layer1"),
    layers.Dropout(rate=DropoutRate),
    layers.Dense(Layer_2_Size, activation=tf.nn.relu, kernel_regularizer=regularizers.L1(l1=L1_a), name="Layer2"),
    layers.Dense(TotalClasses, activation=tf.nn.softmax, name="OutputLayer"), ]

L2_Set1_Layers = [
    layers.Dense(units=Layer_1_Size, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(l2=L2_a[0]), kernel_initializer=tf.keras.initializers.RandomNormal(mean=L1_Mean, stddev=1, seed=seed), name="Layer1"),
    layers.Dense(units=Layer_2_Size, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(l2=L2_a[0]), kernel_initializer=tf.keras.initializers.RandomNormal(mean=L1_Mean, stddev=1, seed=seed), name="Layer2"),
    layers.Dense(units=TotalClasses, activation=tf.nn.softmax, name="OutputLayer")]

L2_Set2_Layers = [
    layers.Dense(units=Layer_1_Size, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(l2=L2_a[1]), kernel_initializer=tf.keras.initializers.RandomNormal(mean=L1_Mean, stddev=1, seed=seed), name="Layer1"),
    layers.Dense(units=Layer_2_Size, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(l2=L2_a[1]), kernel_initializer=tf.keras.initializers.RandomNormal(mean=L1_Mean, stddev=1, seed=seed), name="Layer2"),
    layers.Dense(units=TotalClasses, activation=tf.nn.softmax, name="OutputLayer")]

L2_Set3_Layers = [
    layers.Dense(units=Layer_1_Size, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(l2=L2_a[2]), kernel_initializer=tf.keras.initializers.RandomNormal(mean=L1_Mean, stddev=1, seed=seed), name="Layer1"),
    layers.Dense(units=Layer_2_Size, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(l2=L2_a[2]), kernel_initializer=tf.keras.initializers.RandomNormal(mean=L1_Mean, stddev=1, seed=seed), name="Layer2"),
    layers.Dense(units=TotalClasses, activation=tf.nn.softmax, name="OutputLayer")]

# Create Models
Standard_Model = keras.Sequential(layers=Standard_Layers, name="Standard_Model")
RMSProp1_Model = keras.Sequential(layers=Standard_Layers, name="RMSProp1_Model")
RMSProp2_Model = keras.Sequential(layers=Standard_Layers, name="RMSProp2_Model")
SGD_Standard_Model = keras.Sequential(layers=Standard_Layers, name="SGD_Standard_Model")
SGD_L1_Model = keras.Sequential(layers=L1_Layers, name="SGD_L1_Model")
SGD_L2_Set1_Model = keras.Sequential(layers=L2_Set1_Layers, name="SGD_L2_Set1_Model")
SGD_L2_Set2_Model = keras.Sequential(layers=L2_Set2_Layers, name="SGD_L2_Set2_Model")
SGD_L2_Set3_Model = keras.Sequential(layers=L2_Set3_Layers, name="SGD_L2_Set3_Model")


TrainingTimes = []
# load mnist data
(trainX, trainY), (testX, testY) = mnist.load_data()

# Flatten images to 1D vector of 784 input shape
testX = testX.reshape([-1, InputSize])
trainX = trainX.reshape([-1, InputSize])

# Convert mnist data to float32
# Image value normalization: [0, 255] -> [0, 1]
testX = numpy.array(testX, numpy.float32)
trainX = numpy.array(trainX, numpy.float32)

testX = testX / 255.
trainX = trainX / 255.

# Compile all the models and put them in a list
models_compile()
Models = [Standard_Model, RMSProp1_Model, RMSProp2_Model, SGD_Standard_Model, SGD_L1_Model, SGD_L2_Set1_Model, SGD_L2_Set2_Model, SGD_L2_Set3_Model]
BatchSizesMatrix = [1, 256, numpy.floor(0.8 * trainX.shape[0]).astype(int)]


for i in range(3):
    print("Currently Training: " + Models[0].name + " with Batch Size = " + str(BatchSizesMatrix[i]))
    Timer_Start = time()
    History = Models[0].fit(trainX, trainY, validation_split=ValidationSplit, batch_size=BatchSizesMatrix[i], epochs=Epochs)
    Timer_End = time()
    Total_Time = Timer_End - Timer_Start
    TrainingTimes.append(Total_Time)
    make_plot(History, Models[0].name, BatchSize=BatchSizesMatrix[i])
    clear_session()


for i in range(1, 8):
    print("Currently Training: " + Models[i].name + " with Batch Size = " + str(BatchSizesMatrix[1]))
    Timer_Start = time()
    History = Models[i].fit(trainX, trainY, validation_split=ValidationSplit, batch_size=BatchSizesMatrix[1], epochs=Epochs)
    Timer_End = time()
    Total_Time = Timer_End - Timer_Start
    TrainingTimes.append(Total_Time)
    make_plot(History, Models[i].name, BatchSizesMatrix[1])
    clear_session()


print('=======================================================================================')
print('=====================================RESULTS===========================================')
print('=======================================================================================')
print("Model Name | Batch Size | Training time in seconds")
for i in range(3):
    print(Models[0].name + ' | ' + str(BatchSizesMatrix[i]) + ' | ' + str(TrainingTimes[i]))

for i in range(1, 8):
    print(Models[i].name + ' | ' + str(BatchSizesMatrix[1]) + ' | ' + str(TrainingTimes[i + 2]))
print('=======================================================================================')
print('=======================================================================================')
print('=======================================================================================')
