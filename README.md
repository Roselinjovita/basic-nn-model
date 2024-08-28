# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
The task is to develop a neural network regression model to predict a continuous target variable using a given dataset.
Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error



## Neural Network Model
![Screenshot 2024-08-28 174109](https://github.com/user-attachments/assets/53c93d67-af0e-4e7b-a036-1a2b8d821091)


## DESIGN STEPS

### STEP 1: Loading the dataset

### STEP 2: Split the dataset into training and testing

### STEP 3: Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: Build the Neural Network Model and compile the model.

### STEP 5: Train the model with the training data.

### STEP 6: Plot the performance plot

### STEP 7: Evaluate the model with the testing data.

## PROGRAM
### Name:S.ROSELIN MARY JOVITA
### Register Number:212222230122
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DEEP').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
dataset=df.astype({'input':'int'})
dataset=df.astype({'output':'int'})
df.head()
df.info()
df.describe()
x=dataset[['input']].values
x
y=dataset[['output']].values
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train1=scaler.transform(x_train)
learn=Sequential([
    Dense(units=9,activation='relu',input_shape = [1]),
    Dense(units=9,activation='relu'),
    Dense(units=1)
])
learn.compile(optimizer='rmsprop',loss='mse')
learn.fit(x_train1,y_train,epochs=2000)
loss_df=pd.DataFrame(learn.history.history)
loss_df.plot()
x_test1=scaler.transform(x_test)
learn.evaluate(x_test1,y_test)
x_n1=[[10]]
x_n1_1=scaler.transform(x_n1)
learn.predict(x_n1_1)


```
## Dataset Information

#### df.head()
![Screenshot 2024-08-28 175708](https://github.com/user-attachments/assets/220f5e72-e7c3-495c-995a-cd887b0ae98e)


#### df.info()
![Screenshot 2024-08-28 180907](https://github.com/user-attachments/assets/803e792e-301e-4af1-b940-79d984e64043)


#### df.describe()
![Screenshot 2024-08-28 180919](https://github.com/user-attachments/assets/92713973-3fad-49c7-89f9-fcc4e7a2d9d6)

## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2024-08-28 181327](https://github.com/user-attachments/assets/64927c9d-5b40-4b52-b49b-766e91be8639)



### Test Data Root Mean Squared Error

![Screenshot 2024-08-28 181359](https://github.com/user-attachments/assets/f15c0aab-437c-4351-97ea-e7e68e92b7ee)


### New Sample Data Prediction

![Screenshot 2024-08-28 181443](https://github.com/user-attachments/assets/79d889ba-61bd-49ce-bdd8-cb15b92fafcf)


## RESULT
Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.

