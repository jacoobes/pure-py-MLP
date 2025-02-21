from ucimlrepo import fetch_ucirepo
import pandas as pd
from itertools import chain, combinations
from mlp import Layer, Linear, MultilayerPerceptron, Relu, Sigmoid, Softmax, SquaredError
from sklearn.model_selection import train_test_split



def get_mpg_dataset():
    # fetch dataset
    auto_mpg = fetch_ucirepo(id=9)

    # data (as pandas dataframes)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    # Combine features and target into one DataFrame for easy filtering
    data = pd.concat([X, y], axis=1)

    # Drop rows where the target variable is NaN
    cleaned_data = data.dropna()

    # Split the data back into features (X) and target (y)
    X = cleaned_data.iloc[:, :-1]
    y = cleaned_data.iloc[:, -1]

    # Display the number of rows removed
    rows_removed = len(data) - len(cleaned_data)
    print(f"Rows removed: {rows_removed}")   
    # Do a 70/30 split (e.g., 70% train, 30% other)
    X_train, X_leftover, y_train, y_leftover = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,    # for reproducibility
        shuffle=True,       # whether to shuffle the data before splitting
    )

    # Split the remaining 30% into validation/testing (15%/15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_leftover, y_leftover,
        test_size=0.5,
        random_state=42,
        shuffle=True,
    )

    # Compute statistics for X (features)
    X_mean = X_train.mean(axis=0)  # Mean of each feature
    X_std = X_train.std(axis=0)    # Standard deviation of each feature

    # Standardize X
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Compute statistics for y (targets)
    y_mean = y_train.mean()  # Mean of target
    y_std = y_train.std()    # Standard deviation of target

    # Standardize y
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    return X_train,y_train, X_val, y_val, X_test, y_test

X_train,y_train, X_val, y_val, X_test, y_test = get_mpg_dataset()
activation = Relu()
model = MultilayerPerceptron(
    layers=[
        Layer(fan_in=7, fan_out=5, activation_function=activation),
        Layer(fan_in=5, fan_out=6, activation_function=activation),
        Layer(fan_in=6, fan_out=1, activation_function=activation),
    ]
)

print(X_train.to_numpy().shape)
print(y_train.to_numpy())

loss = SquaredError()
model.train( 
    train_x=X_train.to_numpy(),
    train_y=y_train.to_numpy().reshape(-1, 1),
    val_x=X_val.to_numpy(),
    val_y=y_val.to_numpy().reshape(-1,1),
    loss_func=loss
)


