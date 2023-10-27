import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pytest
import numpy as np
from run import process_data, load_data, train_model  # Import necessary functions


def test_data_loading():
    # Assuming the function to load data is named 'load_data'
    pth = 'data/census_clean.csv'
    data = load_data(pth)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_process_data():
    # Creating a dummy dataset
    data = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'workclass': ['Private', 'Self-emp', 'Private', 'Government'],
        'education': ['Bachelors', 'Masters', 'PhD', 'Bachelors'],
        'salary': ['>50K', '<=50K', '>50K', '<=50K']
    })
    cat_features = ['workclass', 'education']

    # Test the process_data function
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label='salary', training=True)

    # Check if the process_data function encodes correctly and returns the right types

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_model_training():
    # Creating a dummy dataset
    data = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'workclass_Private': [1, 0, 1, 0],
        'education_Bachelors': [1, 1, 0, 1],
        'workclass': ['Private', 'Self-emp', 'Private', 'Government'],
        'salary': ['>50K', '<=50K', '>50K', '<=50K']
    })
    X_train, y_train, _, _ = process_data(data, categorical_features=['workclass'], label='salary', training=True)

    # Test if the model trains successfully
    model = train_model(X_train, y_train)

    assert model

