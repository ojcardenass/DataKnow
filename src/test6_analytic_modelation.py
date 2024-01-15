import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from pathlib import Path

# Read the data
base = Path(__file__).resolve().parent.parent
train = pd.read_csv(base / "Datos3/train.csv")
test = pd.read_csv(base / "Datos3/test.csv")
test_original = test.copy()

# Check the data
# From a first look at the data we can see that there are different types of variables, which include categorical and numerical variables. That means we will have to preprocess the data before we can use it to train the model.
train.head()

# Also, we can see that there are missing values in the data. We will have to deal with them before training the model.
null = train.isnull().sum()
# Now, in order to get a better understanding of the data, we can plot the distribution of the data. An
stats = train.describe()

# Plot the data
# Upon plotting the data, we can see that the data is imbalanced, which means that there are more values of one class than the other.
values = train['FRAUDE'].value_counts()
values = np.array(values)
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.pie(values, labels=['No Fraude', 'Fraude'], autopct='%1.1f%%', shadow=True, startangle=90)
ax.set_title('Distribución de datos')


# Data preprocessing
# We will have to preprocess the data before we can use it to train the model. The preprocessing steps include:
# 1. Drop the columns that are not needed for the model, such as id, FECHA, FECHA_VIN, OFICINA_VIN, and the columns related to distance.
# Drop the columns that are not needed for the model
train = train.drop(['id','Dist_max_NAL','Dist_Sum_INTER','Dist_Mean_INTER','Dist_Max_INTER','Dist_Mean_NAL','Dist_HOY','Dist_sum_NAL','FECHA','FECHA_VIN','OFICINA_VIN'],axis=1)
# 2. Normalize the numerical variables using the standard scaler by removing the mean and scaling to unit variance
cols_to_norm = ['VALOR','INGRESOS','EGRESOS']
train[cols_to_norm] = train[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))
# Drop the rows that have missing values in the column SEGMENTO
train = train.dropna(subset=['SEGMENTO'])

# Encode categorical variables
# 4. Encode the categorical variables using the label encoder from sklearn
types = train.dtypes
obj = train.select_dtypes(include = "object").columns
# Convert the categorical variables to numeric
label_encoder = preprocessing.LabelEncoder()
for label in obj:
    train[label] = label_encoder.fit_transform(train[label].astype(str))

## Defining training data
# For the model to be able to learn, we will have to define the input and output variables. The input variables are all the columns except the output variable, which is FRAUDE.
# Then we will split the data into train and test sets. We will use 80% of the data to train the model and 20% to test it.
# Define the output and input variables
y = train['FRAUDE']
x = train.drop(['FRAUDE'],axis=1)

# Split the data into train and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Build the model with 5 hidden layers
# The model is built using Keras' Sequential API. This allows for the easy creation of a linear stack of layers. The model consists of five dense (fully connected) layers and one dropout layer for regularization. The dense layers use the Rectified Linear Unit (ReLU) activation function, except for the last layer, which uses the sigmoid activation function because it is a binary classification problem.
# The dropout layer helps prevent overfitting by randomly setting a fraction of the input units to zero during training. Overfitting occurs when a model becomes too specialized for the training data and performs poorly on new, unseen data. The dropout layer randomly sets a fraction of input units to zero during training, which helps prevent the model from relying too heavily on any single feature or neuron.

model = Sequential([
    Dense(input_dim = x_train.shape[1], units = 100, activation = 'relu'),
    Dense(units = 60, activation = 'relu'),
    Dropout(0.5),
    Dense(units = 30, activation = 'relu'),
    Dense(units = 10, activation = 'relu'),
    Dense(units = 1, activation = 'sigmoid')
])

# Compile and train the model with Adam optimizer and binary crossentropy loss function
# The model is built using the Adam optimizer and the binary cross-entropy loss function, which are appropriate for binary classification problems. The model is then trained on the training data with a batch size of 10 and for 100 epochs. The loss and accuracy of the training process are plotted to visualize the performance of the model during training.
optimizador = keras.optimizers.Adam(learning_rate=(0.001))
model.compile(optimizer = optimizador, loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size = 10, epochs = 100)

# Plot the training loss and accuracy is expected the loss to decrease and the accuracy to increase with each epoch.
fig, bx = plt.subplots(1,1, figsize=(5,5))
bx.plot(history.history['loss'],color='b')
bx.plot(history.history['accuracy'],color='r')
bx.set_title('Train Loss vs Accuracy', fontsize=12)
bx.legend(['Loss', 'Accuracy'], loc='best', fontsize='x-large')

# Evaluate the model on the test set
score = model.evaluate(x_test,y_test)


# Make predictions on the test set
y_prediction = model.predict(x_test)
test_array = np.array(y_test)
unique, counts = np.unique(test_array, return_counts=True)

# Compute the confusion matrix 
# The confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. The confusion matrix shows the number of correct and incorrect predictions made by the model compared to the actual outcomes.
# The results are accuracy, precision, recall, and F1 score. The accuracy is the proportion of the total number of predictions that were correct. The precision is the proportion of positive predictions that were actually correct. The recall is the proportion of actual positives that were correctly classified. The F1 score is the harmonic mean of precision and recall.
cm = confusion_matrix(test_array,y_prediction.round())
tn, fp, fn, tp = confusion_matrix(test_array,y_prediction.round()).ravel()
result = pd.DataFrame(cm, index=['NO FRAUDE','FRAUDE'],columns=['NO FRAUDE','FRAUDE'])

# Plot the confusion matrix
fig = plt.figure(figsize= (6,6))
ax = fig.add_subplot(1,1,1)
sns.heatmap(result, annot=True, linewidths=1, linecolor= 'white',fmt= 'd', cmap="GnBu")
font= {'family': 'serif',
         'color': 'darkred',
        'weight': 'normal',
        'size':14}
ax.set_title("Matriz de confusion ",fontdict ={'family': 'serif','color': 'black','weight': 'normal','size':40})
ax.set_ylabel("True Label", labelpad=20, fontdict={'family': 'serif','color': 'black','weight': 'normal','size':20})
ax.set_xlabel("Predict Label", labelpad=15, fontdict={'family': 'serif','color': 'black','weight': 'normal','size':20})

# Preprocess the test data
# For evaluating the model on the test data, we will have to preprocess the data in the same way as we did for the training data.
# Fist we will select the columns used for training the model and then we will normalize the numerical variables using the standard scaler.
test = test[train.columns]
cols_to_norm = ['VALOR','INGRESOS','EGRESOS']
test[cols_to_norm] = test[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))
obj = test.select_dtypes(include = "object").columns
label_encoder = preprocessing.LabelEncoder()
for label in obj:
    test[label] = label_encoder.fit_transform(test[label].astype(str))
test = test.drop(['FRAUDE'],axis=1)

# Make predictions on the test data
test_evaluado = model.predict(test)
# Add the predictions to the test dataframe
test_original['FRAUDE'] = test_evaluado.round()

# Save the predictions to an Excel file
test_original.to_excel(base / "output/test_evaluado.xlsx", index = False)