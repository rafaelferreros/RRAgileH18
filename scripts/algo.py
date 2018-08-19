import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def predict(model, data):
    """Predict the result.

    Param: model The model to use.

    Return: The predicted dataset.
    """
    return model.predict(data)


def create_random_forest_regressor():
    """Creates a random forest regressor model.

    Return: A random forest regressor model.
    """
    rfr = RandomForestRegressor(n_estimators = 1000, random_state = 100)
    return rfr


def create_decision_tree_regressor():
    """Creates a decision tree regressor.

    Return: A decision tree regressor model.
    """
    dtr = DecisionTreeRegressor(max_depth=8)
    return dtr


def rmsl_error(real, predicted):
    """Calculates the root mean squared log error.

    Param: real      The real dataset(column).
    Param: predicted The predicted dataset(column).

    Return: The root mean squared log error.
    """
    rmsle = np.sqrt(mean_squared_log_error(real, predicted))
    return round(rmsle, 3)


def train_model(model, data, target, test_percent, random):
    """Train the model.

    Param: model         The model to use.
    Param: data          The dataset to use.
    Param: target        The target dataset (column) to use.
    Param: test_percent: The amount of data from the dataset reserved for
                         testing.
    Param: random:       The random seed to use when splitting the data.
    """
    print('Training the model');
    train_data, test_data, train_target, test_target = train_test_split(
        data,
        target,
        test_size = test_percent,
        random_state = random);

    model.fit(train_data, train_target);

    predictions = predict(model, test_data)
    error = rmsl_error(test_target, predictions)
    print('Model root mean squared log error is:', error)
    print(pd.DataFrame(abs(predictions - test_target)).describe())
    print()
