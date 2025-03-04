import numpy
import tensorflow as tf
import warnings
import seaborn
import keras_tuner
from matplotlib import pyplot as pp
from tensorflow import keras
from keras import backend
from keras import layers
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
print("Process Running...")


def make_plot(ModelHistory, ModelName, BatchSize):
    # Create diagram for accuracy
    pp.plot(ModelHistory.history['accuracy'], color='blue')
    pp.plot(ModelHistory.history['val_accuracy'], color='green')
    pp.title("Accuracy for " + ModelName + " with Batch Size = " + str(BatchSize))
    pp.legend(['Train', 'Validation'], loc='center right')
    pp.xlabel('Epoch')
    pp.ylabel('Accuracy')
    pp.savefig('Accuracy ' + ModelName + " BatchSize=" + str(BatchSize) + ".png")
    print("Plot completed!")
    print("Close the plot window to continue.")
    pp.show()

    # Create diagram for loss
    pp.plot(ModelHistory.history['loss'], color='red')
    pp.plot(ModelHistory.history['val_loss'], color='black')
    pp.title("Loss for " + ModelName + " with Batch Size = " + str(BatchSize))
    pp.legend(['Train', 'Validation'], loc='center right')
    pp.xlabel('Epoch')
    pp.ylabel('Loss')
    pp.savefig('Loss ' + ModelName + "BatchSize=" + str(BatchSize) + ".png")
    print("Plot completed!")
    print("Close the plot window to continue.")
    pp.show()


def best_model_make_plot(BestModelHistory):
    pp.plot(BestModelHistory.history['f1_score'], color='blue')
    pp.plot(BestModelHistory.history['val_f1_score'], color='green')
    pp.title("Training curve for best model")
    pp.legend(['Train F1 score', 'Validation F1 score'], loc='center right')
    pp.xlabel('Iteration')
    pp.ylabel('F1 score')
    pp.savefig('Best model training diagram ' + ".png")
    print("Plot completed!")
    print("Close the plot window to continue.")
    pp.show()


def recall_score(y_actual, y_predicted):
    TruePositives = backend.sum(backend.round(backend.clip(y_actual * y_predicted, 0, 1)))
    PossiblePositives = backend.sum(backend.round(backend.clip(y_actual, 0, 1)))
    result = TruePositives / (PossiblePositives + backend.epsilon())  # kr.epsilon() addition ensures that we don't divide by zero
    return result


def precision_score(y_actual, y_predicted):
    TruePositives = backend.sum(backend.round(backend.clip(y_actual * y_predicted, 0, 1)))
    PossiblePositives = backend.sum(backend.round(backend.clip(y_predicted, 0, 1)))
    result = TruePositives / (PossiblePositives + backend.epsilon())  # kr.epsilon() addition ensures that we don't divide by zero
    return result


def f1_score(y_actual, y_predicted):
    precision = precision_score(y_actual, y_predicted)
    recall = recall_score(y_actual, y_predicted)
    result = 2 * ((precision * recall) / (precision + recall + backend.epsilon()))  # kr.epsilon() addition ensures that we don't divide by zero
    return result


def create_model(mod):
    model = keras.Sequential()
    Layer1_neurons = [64, 128]
    Layer2_neurons = [256, 512]
    LearningRate = [0.1, 0.01, 0.001]
    Layer2_a = [0.1, 0.001, 0.000001]

    L2 = mod.Choice('l2_coeff', values=Layer2_a)
    Layer1Units = mod.Choice('layer1_units', values=Layer1_neurons)
    Layer2Units = mod.Choice('layer2_units', values=Layer2_neurons)
    LearnRate = mod.Choice('learning_rate', values=LearningRate)

    Reg = keras.regularizers.L2(l2=L2)

    Init = keras.initializers.HeNormal()

    NetworkLayers = [
        layers.Dense(units=Layer1Units, activation=tf.nn.relu, kernel_regularizer=Reg, kernel_initializer=Init, name="Layer1", input_shape=(InputSize,)),
        layers.Dense(units=Layer2Units, activation=tf.nn.relu, kernel_regularizer=Reg, kernel_initializer=Init, name="Layer2"),
        layers.Dense(units=10, activation=tf.nn.softmax, name="OutputLayer")]

    for CurrentLayer in NetworkLayers:
        model.add(CurrentLayer)

    model.compile(optimizer=tf.optimizers.SGD(learning_rate=LearnRate), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', precision_score, recall_score, f1_score])
    return model


Epochs = 1000
Patience = 200
BatchSize = 128
TotalClasses = 10
ValidationSplit = 0.2
InputSize = 784  # image shape: 28 * 28 pixels

# load mnist data
(trainX, trainY), (testX, testY) = mnist.load_data()
# Flatten images to 1D vector of 784 input shape
testX = testX.reshape([-1, InputSize])
trainX = trainX.reshape([-1, InputSize])

# Convert mnist data to float32
# Image value normalization: [0, 255] -> [0, 1]
trainX.astype('float32')
testY.astype('float32')
testX.astype('float32')
testX = testX / 255.
trainX = trainX / 255.

testY = to_categorical(testY, TotalClasses)
trainY = to_categorical(trainY, TotalClasses)

Tuner = keras_tuner.Hyperband(create_model, objective=keras_tuner.Objective("f1_score", direction='max'), max_epochs=Epochs, directory='MLP_Network_Tuning', project_name='MLP_Fine_Tuning', overwrite=True)
StopEarly = EarlyStopping(monitor="f1_score", patience=Patience, restore_best_weights=True)
Tuner.search(trainX, trainY, epochs=Epochs, batch_size=BatchSize, validation_split=ValidationSplit, callbacks=[StopEarly],)

Hyperparameters = Tuner.get_best_hyperparameters(num_trials=1)[0]
Layer1Units = Hyperparameters.get('layer1_units')
Layer2Units = Hyperparameters.get('layer2_units')
Learning_Rate = Hyperparameters.get('learning_rate')
Layer2Coefficient = Hyperparameters.get('l2_coeff')

ModelWithTuning = keras.Sequential()

NetworkLayers = [
    layers.Dense(units=Layer1Units, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.L2(l2=Layer2Coefficient), kernel_initializer=keras.initializers.HeNormal(), name="Layer1", input_shape=(InputSize,)),
    layers.Dense(units=Layer2Units, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.L2(l2=Layer2Coefficient), kernel_initializer=keras.initializers.HeNormal(), name="Layer2"),
    layers.Dense(units=10, activation=tf.nn.softmax, name="OutputLayer")]


for CurrentLayer in NetworkLayers:
    ModelWithTuning.add(CurrentLayer)

ModelWithTuning.compile(optimizer=tf.optimizers.SGD(learning_rate=Learning_Rate), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy', precision_score, recall_score, f1_score])

History = ModelWithTuning.fit(trainX, trainY, epochs=200, batch_size=BatchSize, validation_split=ValidationSplit)
best_model_make_plot(History)

PredictedY = ModelWithTuning.predict(testX)
PredictedYLabel = [numpy.argmax(current_predicted) for current_predicted in PredictedY]
testYLabel = [numpy.argmax(current_test) for current_test in testY]
CM = confusion_matrix(testYLabel, PredictedYLabel)

Loss, acc, f1Score, PrecisionScore, RecallScore = ModelWithTuning.evaluate(testX, testY, verbose=0)

print('=======================================================================================')
print('=====================================RESULTS===========================================')
print('=======================================================================================')
print("Hyperparameters:")
print(f"Best Layer 1 size: {Layer1Units}")
print(f"Best Layer 2 size: {Layer2Units}")
print(f"Best Learning Rate: {Learning_Rate}")
print(f"Best L2 Regularization Coefficient: {Layer2Coefficient}")
print('---------------------------------------------------------------------------------------')
print('Best metrics')
print('---------------------------------------------------------------------------------------')
print('Loss:', Loss)
print('Accuracy:', acc)
print('Recall Score:', RecallScore)
print('Precision Score:', PrecisionScore)
print('f1 score:', f1Score)
print('=======================================================================================')
print('=======================================================================================')
print('=======================================================================================')

seaborn.heatmap(CM, annot=True, fmt='d')  # print confusion matrix
pp.title("Confusion Matrix")
pp.figure()
pp.xlabel("Predicted")
pp.ylabel("Actual")
pp.savefig("Confusion Matrix with Tuning.png")
pp.show()
