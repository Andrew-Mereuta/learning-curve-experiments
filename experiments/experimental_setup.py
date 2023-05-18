import sys

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

# Common imports
import openml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor


def get_outer_split(X, y, seed):
    return sklearn.model_selection.train_test_split(X, y, train_size=0.9, random_state=seed, stratify=y)


def get_inner_split(X, y, outer_seed, inner_seed):
    X_learn, X_test, y_learn, y_test = get_outer_split(X, y, outer_seed)
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X_learn, y_learn, train_size=0.9,
                                                                                  random_state=inner_seed,
                                                                                  stratify=y_learn)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def all_elems_equal(s):
    a = s.to_numpy()
    return (a[0] == a).all()


def sqrt_schedule_function(k) -> int:
    """
    Create a schedule according to a geometric sequence of square root of two starting from 16
    :param k: the current increment to compute the training set size
    :return: an integer indicating the training set size
    """
    return int(np.ceil(2.0 ** ((7.0 + k) / 2.0)))


def get_schedule(max_size, schedule_function=sqrt_schedule_function, min_size=16) -> list[int]:
    """
    Create a schedule for increasing the dataset on which a model is evaluated.
    :param max_size: the maximum size of the training set
    :param schedule_function: the scheduling function
    :param min_size: the minimum size of the training set
    :return: a schedule (i.e. a list) of increasing training set sizes.
    """
    res = []
    k = 1.0
    training_size = schedule_function(k)

    while training_size < max_size:
        if training_size >= min_size:
            res.append(training_size)
        k += 1
        training_size = schedule_function(k)

    return res


def split_kfold(X, y, n_splits, random_state) -> list:
    """
    Split the datasets into k stratified folds
    :param X: all instances in the dataset
    :param y: the corresponding label for each instance
    :param n_splits: the number of splits to generate
    :param random_state: seed for splitting reproducibility
    :return: k training and test folds
    """
    skfolds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []

    for train_index, test_index in skfolds.split(X, y):
        X_train_folds = X.iloc[train_index]
        y_train_folds = y.iloc[train_index]
        X_test_folds = X.iloc[test_index]
        y_test_fold = y.iloc[test_index]

        folds.append((X_train_folds, y_train_folds, X_test_folds, y_test_fold))

    return folds


def get_datasets_curve(training_size, X_train_split, y_train_split, X_test_split, y_test_split):
    """
    Given the current training size, divide the current training set into one that has the `training_size` and add
    the unused data to the test set.
    :param training_size: size that the training set needs to have
    :param X_train_split: the current stratified train inputs
    :param y_train_split: the current stratified train labels
    :param X_test_split: the current stratified test inputs
    :param y_test_split: the current stratified test labels
    :return: four splits which conform to the `training_size`
    """
    X_train_k = X_train_split[:training_size]
    y_train_k = y_train_split[:training_size]

    # Append unused data to the test set
    X_test_k = np.append(X_test_split, X_train_split[training_size:], axis=0)
    y_test_k = np.append(y_test_split, y_train_split[training_size:], axis=0)

    return X_train_k, y_train_k, X_test_k, y_test_k


def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]
    # Shuffle
    df = df.sample(frac=1, random_state=42)

    y = np.array(df[ds.default_target_attribute].values)
    # prepare label column as numpy array
    print(f"Read in data frame. Size is {len(df)} x {len(df.columns)}.")
    df = df.drop(columns=[ds.default_target_attribute])
    cat_attributes = list(df.select_dtypes(include=['category', 'object', 'bool']))
    X = pd.get_dummies(df, columns=cat_attributes, dtype=int)

    # pca = PCA(n_components=0.5)
    # X = pca.fit_transform(X)

    if y.dtype != int:
        y_int = np.zeros(len(y)).astype(int)
        vals = np.unique(y)
        for i, val in enumerate(vals):
            mask = y == val
            y_int[mask] = i
        y = pd.Series(y_int)

    print(f"Data is of shape {X.shape}.")
    return X, y


class Experiment:
    """
    A class for learning curve experiments, which runs each dataset against each learner and generate learning
    curves for increasing dataset size according to a schedule given by the `schedule_function`. The results are
    all saved to a file called `experiment_results.gz`.
    """

    def __init__(self, datasets, learners, tuning_params=None, tuning_strategy=None, n_splits=10,
                 performance_metric=accuracy_score, schedule_function=sqrt_schedule_function, random_state=42):
        """
        Create a new class for experiments, which runs each dataset against each learner and generate learning curves
        for increasing dataset size according to a schedule given by the `schedule_function`. The results are all saved
        to a file called `experiment_results.gz`.

        :param datasets: a list of id for datasets which are on OpenML
        :param learners: a list of instantiated learners
        :param tuning_params: a list of dictionaries for tuning parameter, each of which corresponding a learner
        of the same index, default is `None` if hyperparameters should not be tuned
        :param tuning_strategy: an uninstantiated class of the strategy for tuning (e.g. `RandomSearchCV`), default is
        `None` if hyperparameters should not be tuned (only one class should be passed into this parameter)
        :param n_splits: the number of splits to generate which corresponds to the number of learning curves generated
        for each dataset and classifier
        :param performance_metric: the performance metric for the learner's predictions
        :param schedule_function: the function to schedule the increase of the training set size
        :param random_state: random seed
        """
        self.datasets: list[int] = datasets
        self.learners: list = learners
        self.tuning_params = tuning_params
        self.tuning_strategy = tuning_strategy
        self.n_splits: int = n_splits
        self.performance_metric = performance_metric
        self.schedule_function = schedule_function
        self.random_state = random_state

        # If either the tuning_params or tuning_strategy is None/empty then create an array of None for tuning params
        if tuning_params is None or tuning_strategy is None or len(tuning_params) == 0:
            self.tuning_params = [None] * len(learners)

    def __preprocess_data(self, df):
        df_num = df.select_dtypes(include=[np.number])
        num_attribs = list(df_num)

        # build num pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('minmax_scaler', MinMaxScaler()),
        ])

        # build full pipeline
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
        ])

        full_pipeline.fit(df)

        return full_pipeline

    def __evaluate(self, openmlid, X, y, og_learner):
        file = '../lcdb-orig/3-and-accuracy.csv'
        config = pd.read_csv(file)

        prediction_table = []

        for i, row in config.iterrows():
            size_train = row['size_train']  # anchor
            size_test = row['size_test']
            outer_seed = row['outer_seed']
            inner_seed = row['inner_seed']
            X_train, X_valid, X_test, y_train, y_valid, y_test = get_inner_split(X, y, outer_seed, inner_seed)
            X_train = X_train[:size_train]
            y_train = y_train[:size_train]

            learner = clone(og_learner)
            learner.fit(X_train, y_train)

            y_hat_train = learner.predict(X_train)
            y_hat_valid = learner.predict(X_valid)
            y_hat_test = learner.predict(X_test)

            accuracy_train = sklearn.metrics.accuracy_score(y_train, y_hat_train)
            accuracy_valid = sklearn.metrics.accuracy_score(y_valid, y_hat_valid)
            accuracy_test = sklearn.metrics.accuracy_score(y_test, y_hat_test)

            prediction_table.append((og_learner.__class__.__name__, openmlid, size_train, size_test,
                                     outer_seed, inner_seed, accuracy_train, accuracy_valid, accuracy_test))

        return prediction_table

    def run_all_experiments(self):
        """
        Generate results for each dataset and for each learner and save it to "experiment_results.gz"
        """

        for openmlid in self.datasets:
            results = []
            X, y = get_dataset(openmlid)
            for learner_index in range(0, len(self.learners)):
                learner = self.learners[learner_index]
                hyperparameters = self.tuning_params[learner_index]
                results = results + self.__evaluate(openmlid, X, y, learner)
                print(str(openmlid) + ' trained on ' + str(learner) + ' done')
            df_results = pd.DataFrame(results,
                                      columns=['learner', 'openmlid', 'size_train', 'size_test',
                                               'outer_seed', 'inner_seed', 'score_train', 'score_valid', 'score_test'])
            df_results.to_pickle('../data/experiment/' + str(openmlid) + '_results.gz')


if __name__ == '__main__':
    # Regression datasets
    diamonds = 42225
    us_crime = 315
    houses = 537
    abalone = 42726
    cpu_small = 562
    kin8nm = 189
    sulfur = 23515
    elevators = 216
    house_8L = 218
    wind = 503

    datasets = [3]  # diamonds, us_crime, houses, abalone, cpu_small, kin8nm, sulfur, elevators, house_8L, wind]

    # Number of splits
    n_splits = 25

    # Learners
    learners = [LinearSVC()]
    # SGDRegressor(max_iter=100000), , DecisionTreeRegressor(),
    #         GradientBoostingRegressor(), SVR()]

    # Create a new instance of an experiment
    e = Experiment(datasets, learners, performance_metric=mean_squared_error, n_splits=n_splits)

    print("Running experiments...")
    e.run_all_experiments()
    print("Finished running all experiments!")
