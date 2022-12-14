train_y_current = train[1:train.shape[0]]
train_y_past = train[0:train.shape[0] - 1]
train_y_features = train_y_current.reset_index(drop=True) - train_y_past.reset_index(drop=True)
train_x_features = []

# Make transformations for test df
test_y_current = test[1:test.shape[0]]
test_y_past = test[0:test.shape[0] - 1]
test_y_features = test_y_current.reset_index(drop=True) - test_y_past.reset_index(drop=True)
test_x_features = []

num_days_year = 253
start_t = int(num_days_year / 2)
num_features = int(0.2 * start_t)

# Fill out train_x_features
for i in range(train_y_features.shape[0] - num_features):
    current_i = i + num_features
    x_feature = []
    for j in range(num_features):
        x_feature.append(train_y_features[current_i - j - 1])
    train_x_features.append(x_feature[::-1])

# Fill out test_x_features
for i in range(test_y_features.shape[0] - num_features):
    current_i = i + num_features
    x_feature = []
    for j in range(num_features):
        x_feature.append(test_y_features[current_i - j - 1])
    train_x_features.append(x_feature[::-1])