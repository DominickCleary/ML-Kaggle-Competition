# ML-Kaggle-Competition

---

## About

The code runs through this process:

* Load the csv files
* Clean the data
  * Fix data representing the same value but different names. e.g `"0"` and `"No"` in `"Hair Color"`
  * Unknown numerical values were replaced with the mean
  * Missing categorical data became "unknown"
* Encode the data
  * I used one-hot encoding for `"Gender"` and `"Hair Colour"`
  * A binary encoder for `"Profession"` and `"Country"` as I found the cardinality too high for one-hot
  * I used ordinal encoding for `"University Degree"` as I found it to be slightly more accurate
* Split the data
  * I split the data 80-20
* Scaling
  * I used the standard scaler to fit all the data
* Other
  * `"Wears Glasses"` was irrelevant to the result so it was dropped
  * Outliers with a z score > 3 were removed
* Model
  * After trying models such as SGD and SVM i found linear regression to be the most accurate.

Privately my rmse is ~76'000 but publicly it scored only ~120'000
