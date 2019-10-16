import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats

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


def preprocess_data(df):
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

    # q = df["Income in EUR"].quantile(0.75)
    # df = df[df["Income in EUR"] < q]

    num_df = df.select_dtypes(include=["number"])
    cat_df = df.select_dtypes(exclude=["number"])

    idx = np.all(stats.zscore(num_df) < 3, axis=1)

    df = pd.concat([num_df.loc[idx], cat_df.loc[idx]], axis=1)

    df = pd.get_dummies(
        df, columns=[GENDER, HAIR_COLOR], drop_first=True)

    ce_bin = ce.BinaryEncoder(cols=[PROFESSION, COUNTRY])
    df = ce_bin.fit_transform(df)

    return df


def process_data(df):
    X = df.loc[:, df.columns != INCOME]
    X = X.loc[:, X.columns != "Instance"]
    X = X.loc[:, X.columns != WEARS_GLASSES]
    # X = X.iloc[:, 1:].values
    # X = df[[YEAR_OF_RECORD, AGE, SIZE_OF_CITY, BODY_HEIGHT]].values
    y = df[INCOME]

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
    # y_pred = np.abs(y_pred)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.sample(25)

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, y_pred)))


def main():
    df = read_data(
        "csv/tcd ml 2019-20 income prediction training (with labels).csv")
    df = preprocess_data(df)
    process_data(df)
    # data = pd.DataFrame({"X": df[WEARS_GLASSES], "Y": df[INCOME]})
    # data = data.head(1000)
    # data.plot(x="X", y="Y", kind="scatter", figsize=(10, 8))
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # plt.show()


if __name__ == "__main__":
    main()
