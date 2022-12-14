# timeseries_forecasting
This is a package for time series forecasting.

The complete documentation to this package will be uploaded soon. 

name: time_series_forecasting
version: 1.0.0
dependencies:
  - numpy
  - pandas
  - scikit-learn
  - sklearn.SGDRegressor
  - sklearn.RandomForestRegressor
  - keras.LSTM
  - statsmodels.ARIMA
  - statsmodels.VAR
  - statsmodels.Theta
defaults:
  arima:
    p: 5
    d: 0
    q: 0
  random_forest_regressor:
    n_estimators: 100
    bootstrap: false
  var:
    maxlags: 3
  theta:
    deseasonalize: false
  sgdr:
    learning_rate: adaptive
    num_features: 25
  lstm:
    activation: relu
    optimizer: adam
    loss: mean_squared_error
    time_steps: 25
    num_features: 1
    layer1_units: 10
    layer2_units: 5
    layer3_units: 5
    output_layer_units: 1
input_path: /datasets/test_timeseries.csv
output_path: /predictions


