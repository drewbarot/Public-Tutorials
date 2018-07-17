"""
Machine Learning for Stock Market Predictions in 25 Lines
@author: Drew

Hello! This tutorial is meant as a walkthrough for you to learn a bit about
how we can apply some great libraries that are freely available to do some
basic work in predicting the movements of stock prices. 

Over the course of this tutorial, we'll be trying to predict whether in the
next time step, a certain product will go up or down. To keep things simple,
we'll use the S&P 500 product, SPX. For features, we'll stick with the 
Open, High, Low, and Close prices, and volume. 

We'll use an awesome library called XGBoost (XGB) to do most of the heavy 
lifting ML stuff, as well as a few other libraries to do some data processing.
XGBoost is an awesome library that's used a lot in competitions for ML and 
data mining online, and for good reason. It has awesome performance, is
lightning fast, and is somewhat easier to work with an understand, relative
to other machine learning models like Neural Network. It uses decision trees
and a process called boosting, that iteratively improves the performance of 
a decision tree based model. 
Check below for a list of libraries we'll use. 

It's important to note that this tutorial is not something you should deploy
to start trading your hard earned cash, and the performance is pretty poor.
Each product trades differently, and requires different considerations. It is
far more important to consider what features to use, and how to use them, than
it is worrying about the model itself. Also, as with any model, it is almost
impossible to gain 100% accuracy, especially in trading, where strategies 
always adapt. 

With that said, you'll need these packages/versions:
    Python 3.5
    pandas 0.22.0
    NumPy 1.14.2
    Scikit-Learn 0.19.1
    XGBoost 0.71
"""
# Import the libraries we need
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as sk


'''
Data Import and Preprocessing

Our first step is to bring in our data, and convert it to a format that we can
interpret and manipulate within Python, specifically with the Pandas library. 
We have a CSV of OHLC prices and Volume that I've pulled from online. As long 
as you can download daily OHLC and Volume going back a few years, you should
have some basic data to start training your model. Here we're using OHLC and 
Volume as our features, just it's really important to choose the best available
features to you if you ever want to improve your model's performance. You can
also apply modifications to your data if you want to further improve your
models accuracy as well. We'll engineer a very basic feature here, which is
the return of close prices from day to day. This model will perform poorly as
we are using raw values of OHLC and Volume (which tend to increase overtime),
and as such, we won't make meaningful relations from the data - but this is
where you have to consider, what features make sense. As long as you can
understand the model, and make great features, you can do well!

A few tricks in case the read_csv function doesn't work are:
    Make sure the file is a .csv
    Specify the path to the file as I have, with an r in the fron before "
    Place the file in the same folder as your Python file and remove the C:\...
'''
# Data is pulled from a file 
data = pd.read_csv(r"C:\Users\Drew\Downloads\Tutorials\SPX.csv", index_col=0)
data["Close Return"] = data["Close"].pct_change()
# Convert the index to date time format, so that it is easier to work with
data.index = pd.to_datetime(data.index)


'''
Cleaning and Labelling

Now that we have our data, the next step is to work on some really basic
cleaning and labelling. Here, we'll just drop any rows that have a missing, or
NaN value, and keep only those that have all values available. It's important
to note that you can have missing values in a boosted tree model when using 
XGBoost, but it's poor practise as we cannot use this same data if we want to
use something like a neural network. 

We're also going to assign a label. A label is just something that is the 
'correct value,' that we're trying to predict. Here, we know that if the next
week's price is higher, we'll want to buy this week. If it's lower, we should
sell. Labels can get a lot more complicated than this, but we're just going to
do this simply. We'll also save our raw data to a CSV in case we ever want to 
use it for another model. 
'''
# Drop any NaN rows
data= data.dropna(how="any")
#Label assigned, 0 for sell, and 1 for buy
data["Label"] = np.where(data["Close"] > (data["Close"].shift(-1)), 0, 1)
# Data saved to a CSV in our directory
pd.DataFrame.to_csv(data, "SPX_data.csv", index=False)  

'''
Defining Train, Test and Normalizing

A machine learning model needs a lot of data to effectively train and make
meaningful predictions, and some data to test the results of training on. When
we pass in our data, we split it into a train set and a test set. The train 
will be used to train and improve the model through each boosting iteration,
and the test set will be used later to evaluate. 

Although you don't need to normalize, we're going to do it here as it is an
important process for almost all other ML applications. Normalizing simply
takes your data and scales it to a normal range. Usually, this range is
between 0 and 1, however, it can also be from -1 to 1. We set our normalizing
parameters based on the Std dev and mean of our training set, so we may get
values above 1 in our test set. There are numerous reasons for this, which I
won't get into. We can use a library called Scikit-learn for this, which makes
things really easy. Here, we assign our maximum value 1, and minimum 0. 

Lastly, we'll want to create an X and y component of both train and test. In 
both, the X component is the data our model can see (OHLC and Volume), and the
y label is the data our model is supposed to try and predict. We'll split the
data by dropping the label for X, and only including the label for y. We also
want to convert this into a DMatrix format, a specific data structure that XGB
can read quickly. This step is exclusive to XGB, and not required in other ML
approaches. 
'''
# Defining train and test
train_data = data['1992-01-01':'2004-12-31'].copy()
test_data = data['2005-01-01':'2006-01-01'].copy()
# Initializing a normalizer from Scikit-Learn, called sk here
scaler = sk.MinMaxScaler()
scaler.fit(train_data)
# Normalizing the data
train_data.loc[:, train_data.columns] = scaler.transform(train_data)
test_data.loc[:,test_data.columns] = scaler.transform(test_data)
# Train and Test Data and Label Assignment
X_train = train_data.drop("Label",1)
y_train = train_data["Label"]
X_test = test_data.drop("Label",1)
y_test = test_data["Label"]
# Split into DMatrix, format of XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

'''
Machine Learning

Now that we've processed some of our data, we can finally start with the ML,
what we've been waiting for!

We're using the full Python version of XGB, but you can also use the 
Scikit-Learn version if you're more comfortable with that. 

First, we'll define some basic parameters. Parameters are just some basic rules
and instructions for our model to follow while it's fitting the model, all to
make sure that it trains how we want it to. We're specifying that the model
shouldn't go deeper than 5 layers (it can go less than this), that that it 
should use a learning rate of 0.1 to conduct gradient descent. It's also
going to be using binary logistic classification for it's objective. 

Next, we specify the number of rounds. The more rounds, the more overfit the
model, but too few means not enough learning was done. Each round is an
iterative improvement of a previous tree model, and helps our model learn. We
will use 100 rounds here. It's like epochs in a neural network!

Then, the big show! We'll train our model by passing in our parameters, our
train data, and for a specified number of rounds. This will take a few seconds
to a few minutes to process depending on how much data you've pulled. This 
trained model can then be used to predict output labels based on our inputs in
the dtrain matrix. 

After we've done our predictions, we'll make an array with them. The numbers 
won't come in as clean 0 and 1 values as the tree cannot perfectly fit them. It
will ouput the number close to what it expects to be the real value, so we
can round anything above 0.5 to 1, and anything below to 0, as our buy and sell
signals respectively. 

After we've done this, we can plug it into an accuracy measure which will
benchmark how accurate we were in our predictions!
'''
# Parameters for this model. Maximum of 5 layers, learning rate of 0.1, and 
# binary logistic classification which is used for two class predictions
param = {'max_depth':5, 'eta':0.1, 'objective':'binary:logistic'}
# Go through 100 rounds of boosting
iters = 100
# Create and train the model given our parameters, training set, and rounds
model = xgb.train(param, dtrain, iters)
# Create predictions with out test set from the model we created
preds = model.predict(dtest)
# Round the numbers off to 0 and 1 for buy and sell
best_preds = np.asarray([np.round(line) for line in preds])
# Test the accuracy of our model against the true values
accuracytest = accuracy = (accuracy_score(y_test, best_preds))
# Print the accuracy
print("Accuracy on test set was {0}%. The tree went through {1} iterations" 
      .format(accuracytest*100, iters))

'''
Output: Accuracy on test set was 49.2%. The tree went through 100 iterations
'''

'''
Done! In 25 lines of code and about 175 lines of comments/imports. 

Congratulations! You've succesfully trained and tested a stock prediction 
algorithm using XGBoost. Keep in mind, this is just one way of tackling the 
problem of stock market predictions, and there are countless ways to approach
it. This tutorial shouldn't be used to make investment decisions, but can
be a great resource to start exploring more complex and accurate models. 

If you're interested, check out the XGB site at 
https://xgboost.readthedocs.io/en/latest/python/python_intro.html. It offers
some great tips and tricks, and can walk you through what we've done here in
a more general sense.
'''
