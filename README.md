# pkg_pyknnclassifier

<img src="img/logo.png" width="300" alt="pkg_pyknnclassifier logo">

A k-Nearest Neighbors (KNN) classifier for Python.

## About
Our package, named "pkg_pyknnclassifier," is a comprehensive toolkit for k-Nearest Neighbors (k-NN) modeling and evaluation. It offers a set of functions designed to facilitate various aspects of working with k-NN algorithms, from loading the data, calculating distances to making predictions and assessing model performance. We aim to simplify the process by providing essential functionalities for data manipulation, model evaluation, and scaling.

This pacakge consists of five functions and are explained as below:
- calculate_distance(obs_1, obs_2): This function calculates the Euclidean distance between two observations for the KNN model to find the similarity score.
- predict(unlabel_df): This function predicts the labels of the unlabled observations based on the similarity score calculated from Euclidean distance.
- evaluate(y_true, y_pred, metric='accuracy'): This function calculates evaluation metrics such as accuracy, precision, recall, and F1 score for a k-NN model based on true labels and predicted labels.
- data_loading(str_of_path): This function takes in a file path and load the data
- scaling(df, impute_strategy, scale_method): This function allows user to choose the method of data imputation and scaling, and apply to the data.

## Installation

```bash
$ pip install pkg_pyknnclassifier
```

## Usage

- TODO

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pkg_pyknnclassifier` was created by "Bill Wan, Sho Inagaki, Shizhe Zhang, Weiran Zhao". It is licensed under the terms of the MIT license.

## Credits

`pkg_pyknnclassifier` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
