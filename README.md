# pkg_pyknnclassifier

<img src="img/logo.png" width="300" alt="pkg_pyknnclassifier logo">

## 📄 About
Our package, named "pkg_pyknnclassifier," is a comprehensive toolkit for k-Nearest Neighbors (k-NN) modeling and evaluation. It offers a set of functions designed to facilitate various aspects of working with k-NN algorithms, from loading the data, calculating distances to making predictions and assessing model performance. We aim to simplify the process by providing essential functionalities for data manipulation, model evaluation, and scaling.

## 📦 Functions
This pacakge consists of five functions and are explained as below:
- data_loading(str_of_path, target_column): This function loads data from a file path and split into features and target.
- scaling(df, impute_strategy, scale_method): This function allows user to choose the method of data imputation and scaling, and apply to the data.
- calculate_distance(obs_1, obs_2, method = "Euclidean"): This function calculates the Euclidean distance between two observations for the KNN model to find the similarity score.
- find_neighbors(labeled_arraies, unlabeled_array, k): This function finds the indices of the 'k' nearest neighbors in a collection of labeled arrays to a given unlabeled array. 
- predict(train_X, train_y, unlabel_df, pred_method, k): This function predicts the labels of the unlabled observations based on the similarity score calculated from Euclidean distance.
- evaluate(y_true, y_pred, metric='accuracy'): This function calculates evaluation metrics such as accuracy, precision, recall, and F1 score for a k-NN model based on true labels and predicted labels.


## 🛠️ Installation
To sucessfully run the following commands of installation, we would need `conda` and `poetry`, guide included in the link ([conda](https://docs.conda.io/projects/miniconda/en/latest/), [poetry](https://python-poetry.org/docs/))

1. Direct to the root of this repository
2. Create a  virtual environment in Conda with Python by the following commands at terminal and activate it:
```bash
$ conda create --name pyknnclassifier python=3.11 -y
```
```bash
$ conda activate pyknnclassifier
```
3. Install this package via poetry, run the following command. 
```bash
$ poetry install
```

## ✅ Testing
To test this package, please run the following command from the root directory of the repository:
```bash
pytest tests/
```
- branch coverage could be viewed with the following command:
```bash
pytest --cov-branch --cov=pkg_pyknnclassifier
```

## Usage

- TODO

## 📚 Package Integration within the Python Ecosystem
`pkg_pyknnclassifier`, while acknowledging the robustness and the capabilities of [scikit-learn's KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), aims to offer a specialized and streamlined toolkit tailored explicitly for k-Nearest Neighbors classification tasks. As a lightweight and focused alternative, `pkg_pyknnclassifier` serves users who seek a concise package that offers calculating distances, making predictions, and evaluating k-NN models functions. While scikit-learn covers a broader spectrum of machine learning algorithms, `pkg_pyknnclassifier` provides a more specialized package, potentially appealing to those who prefer a tailored implementation of their k-NN workflows. 

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## 📜 License

`pkg_pyknnclassifier` was created by "Bill Wan, Sho Inagaki, Shizhe Zhang, Weiran Zhao". It is licensed under the terms of the MIT license.

## 📚 Credits

`pkg_pyknnclassifier` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
