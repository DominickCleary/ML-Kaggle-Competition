import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.impute import MissingIndicator
from sklearn.svm import SVR
from sklearn import metrics
from scipy import stats
from csv import writer

YEAR_OF_RECORD = "Year of Record"
GENDER = "Gender"
AGE = "Age"
COUNTRY = "Country"
SIZE_OF_CITY = "Size of City"
PROFESSION = "Profession"
DEGREE = "University Degree"
WEARS_GLASSES = "Wears Glasses"
HAIR_COLOR = "Hair Color"
BODY_HEIGHT = "Body Height [cm]"
INCOME = "Income in EUR"


def read_data(filename):
    return pd.read_csv(os.path.join(os.path.dirname(__file__), filename))


def preprocess_data(df, is_training):
    df.dropna(axis=0, thresh=1, inplace=True)
    df.reset_index(inplace=True, drop=True)

    df[AGE].fillna(df[AGE].mean(), inplace=True)
    df[YEAR_OF_RECORD].fillna(df[YEAR_OF_RECORD].mean(), inplace=True)
    df.fillna(value="unknown", inplace=True)

    df[GENDER].replace("0", "unknown", inplace=True)

    df[DEGREE].replace("0", "No", inplace=True)

    df[HAIR_COLOR].replace("Unknown", "unknown", inplace=True)
    df[HAIR_COLOR].replace("0", "unknown", inplace=True)

    cat = pd.Categorical(df[DEGREE], categories=[
                         "unknown", "No", "Bachelor", "Master", "PhD"], ordered=True)
    labels, np.NaN = pd.factorize(cat, sort=True)
    df[DEGREE] = labels
    df[AGE] = pd.to_numeric(df[AGE].astype(int), downcast="unsigned")
    df[YEAR_OF_RECORD] = pd.to_numeric(
        df[YEAR_OF_RECORD].astype(int), downcast="unsigned")
    df[SIZE_OF_CITY] = pd.to_numeric(
        df[SIZE_OF_CITY].astype(int), downcast="unsigned")
    df[BODY_HEIGHT] = pd.to_numeric(
        df[BODY_HEIGHT].astype(int), downcast="unsigned")
    df[WEARS_GLASSES] = df[WEARS_GLASSES].astype(bool)

    if is_training:
        # q = df["Income in EUR"].quantile(0.97)
        # df = df[df["Income in EUR"] < q]
        num_df = df.select_dtypes(include=["number"])
        # cat_df = df.select_dtypes(exclude=["number"])

        idx = np.all(stats.zscore(num_df) < 3, axis=1)

        # df = pd.concat([num_df.loc[idx], cat_df.loc[idx]], axis=1)
        df = df[idx]
        print(df.head())

    return df


def process_data(testing, training):
    X_train = training.loc[:, training.columns != INCOME]
    X_train = X_train.loc[:, X_train.columns != "Instance"]
    X = X.loc[:, X.columns != WEARS_GLASSES]
    # X = X.iloc[:, 1:].values
    # X = df[[YEAR_OF_RECORD, AGE, SIZE_OF_CITY, BODY_HEIGHT]].values
    y_train = training[INCOME]

    X_test = testing.loc[:, testing.columns != INCOME]
    X_test = X_test.loc[:, X_test.columns != "Instance"]
    X = X.loc[:, X.columns != WEARS_GLASSES]
    # print(X_test.head())

    # for i in range(37):
    # print(X_test.iloc[:, 25])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # regressor = SGDRegressor(alpha=0.1, n_iter_no_change=100);
    # regressor.fit(X_train, y_train)

    # regressor = SVR(kernel = "poly", epsilon=1, verbose=True, gamma="auto")
    # regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    print(y_pred)

    output = pd.read_csv(os.path.join(os.path.dirname(
        __file__), "csv/tcd ml 2019-20 income prediction submission file.csv"))
    output["Income"] = y_pred
    output.to_csv(os.path.join(os.path.dirname(
        __file__), "csv/tcd ml 2019-20 income prediction submission file.csv"), index=False)
    # print(output.head(25))

    # print(y_pred.shape)


def test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # regressor = SGDRegressor(alpha=0.5, n_iter_no_change=20);
    # regressor.fit(X_train, y_train)

    # regressor = SVR(kernel = "poly", epsilon=1, verbose=True, gamma="auto")
    # regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    # output = pd.read_csv(os.path.join(os.path.dirname(
    #     __file__), "tcd ml 2019-20 income prediction submission file.csv"))
    # output["Income"] = y_pred
    # output.to_csv(os.path.join(os.path.dirname(
    #     __file__), "tcd ml 2019-20 income prediction submission file.csv"), index=False)
    # print(output.head(25))
    print(y_pred)


def main():
    cat_columns = [GENDER, HAIR_COLOR, WEARS_GLASSES, PROFESSION, COUNTRY]
    training = read_data(
        "tcd ml 2019-20 income prediction training (with labels).csv")
    training = preprocess_data(training, True)
    # print(training.head())

    testing = read_data(
        "tcd ml 2019-20 income prediction test (without labels).csv")
    testing = preprocess_data(testing, False)

    training = pd.get_dummies(
        training, columns=[GENDER, HAIR_COLOR, WEARS_GLASSES])

    ce_bin = ce.BinaryEncoder(cols=[PROFESSION, COUNTRY])
    training = ce_bin.fit_transform(training)

    cat_dummies = [
        col for col in training
        if "_" in col
        and col.split("_")[0] in cat_columns]
    # print(training.dtypes)

    # print(cat_dummies)

    processed_cols = list(training.columns[:])

    # testing = pd.get_dummies(testing, columns=cat_columns)
    testing = pd.get_dummies(
        testing, columns=[GENDER, HAIR_COLOR, WEARS_GLASSES])

    testing = ce_bin.fit_transform(testing)

    for col in testing.columns:
        if ("_" in col) and (col.split("_")[0] in cat_columns) and col not in cat_dummies:
            print("Removing additional feature {}".format(col))
            testing.drop(col, axis=1, inplace=True)

    for col in cat_dummies:
        if col not in testing.columns:
            print("Adding missing feature {}".format(col))
            testing[col] = 0

    X = training.loc[:, training.columns != INCOME]
    X = X.loc[:, X.columns != "Instance"]

    y = training[INCOME]

    process_data(testing, training)
    # test(X, y)


if __name__ == "__main__":
    main()
