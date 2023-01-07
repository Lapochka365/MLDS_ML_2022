import os
import pickle
import warnings
from datetime import datetime
from typing import Union

import numpy as np
import optuna
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def replace_torque_for_kgm(df_with_kgm_torque: pd.Series) -> pd.DataFrame:
    df_with_kgm_torque = df_with_kgm_torque.str.extractall(r"([\d\.,-]+)").reset_index(
        level=1
    )
    df_with_kgm_torque["torque"] = df_with_kgm_torque[df_with_kgm_torque["match"] == 0][
        0
    ]
    df_with_kgm_torque["max_torque_rpm"] = df_with_kgm_torque[
        df_with_kgm_torque["match"] == 1
    ][0]

    df_with_kgm_torque = df_with_kgm_torque.drop(columns=["match", 0])
    df_with_kgm_torque = df_with_kgm_torque.reset_index()
    df_with_kgm_torque = df_with_kgm_torque.drop_duplicates(subset="index")
    df_with_kgm_torque.index = df_with_kgm_torque["index"]
    df_with_kgm_torque.index.name = None
    df_with_kgm_torque = df_with_kgm_torque.drop(columns="index")

    df_with_kgm_torque.loc[:, "torque"] = (
        df_with_kgm_torque["torque"].astype("float") * 9.8
    )
    df_with_kgm_torque.loc[:, "max_torque_rpm"] = (
        df_with_kgm_torque["max_torque_rpm"]
        .apply(
            lambda x: x.split("-")[1].replace(",", "")
            if "-" in str(x)
            else str(x).replace(",", "")
        )
        .astype("float")
    )

    return df_with_kgm_torque


def replace_torque_for_nm(df_with_nm_torque: pd.Series) -> pd.DataFrame:
    df_with_nm_torque = df_with_nm_torque.str.extractall(r"([\d\.,-]+)").reset_index(
        level=1
    )
    df_with_nm_torque["torque"] = df_with_nm_torque[df_with_nm_torque["match"] == 0][0]
    df_with_nm_torque["max_torque_rpm"] = df_with_nm_torque[
        df_with_nm_torque["match"] == 1
    ][0]

    df_with_nm_torque = df_with_nm_torque.drop(columns=["match", 0])
    df_with_nm_torque = df_with_nm_torque.reset_index()
    df_with_nm_torque = df_with_nm_torque.drop_duplicates(subset="index")

    df_with_nm_torque.index = df_with_nm_torque["index"]
    df_with_nm_torque.index.name = None
    df_with_nm_torque = df_with_nm_torque.drop(columns="index")

    df_with_nm_torque.loc[:, "torque"] = df_with_nm_torque["torque"].astype("float")
    df_with_nm_torque.loc[:, "max_torque_rpm"] = (
        df_with_nm_torque["max_torque_rpm"]
        .apply(
            lambda x: x.split("-")[1].replace(",", "")
            if "-" in str(x)
            else str(x).replace(",", "")
        )
        .astype("float")
    )

    return df_with_nm_torque


class Pipeline:
    def __init__(
        self,
        train_path: str,
        test_path: str,
        categorical_cols: list,
        numeric_cols_w_nan: list,
        alpha_start: float,
        alpha_end: float,
        random_state_start: int,
        random_state_end: int,
        solvers: list,
        number_of_trials: int,
        save_path: str,
        cv: int,
    ):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.cat_columns = categorical_cols
        self.numeric_columns = numeric_cols_w_nan
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.random_state_start = random_state_start
        self.random_state_end = random_state_end
        self.solvers = solvers
        self.n_trials = number_of_trials
        self.save_path = save_path
        self.saved = False
        self.n_splits = cv

    def _create_features_from_torque(
        self, samples: pd.DataFrame = pd.DataFrame()
    ) -> Union[None, pd.DataFrame]:
        datasets = [self.train_df, self.test_df] if samples.empty else [samples]
        for dataset in datasets:
            if "torque" in dataset.columns:
                kgm_torque = dataset["torque"][
                    dataset["torque"]
                    .astype("str")
                    .str.contains("kgm", case=False, regex=False)
                ]
                kgm_torque_index = kgm_torque.index
                processed_kgm_torque = replace_torque_for_kgm(kgm_torque)

                nm_torque = dataset["torque"].loc[~dataset.index.isin(kgm_torque_index)]
                nm_torque_index = nm_torque.index
                processed_nm_torque = replace_torque_for_nm(nm_torque)

                dataset["max_torque_rpm"] = np.NaN
                dataset.loc[:, "torque"] = np.NaN

                dataset.loc[kgm_torque_index, "torque"] = processed_kgm_torque["torque"]
                dataset.loc[kgm_torque_index, "max_torque_rpm"] = processed_kgm_torque[
                    "max_torque_rpm"
                ]

                dataset.loc[nm_torque_index, "torque"] = processed_nm_torque["torque"]
                dataset.loc[nm_torque_index, "max_torque_rpm"] = processed_nm_torque[
                    "max_torque_rpm"
                ]
        if not samples.empty:
            return datasets[0]

    def _fill_na(self, samples: pd.DataFrame = pd.DataFrame()) -> Union[None, np.array]:
        if not samples.empty:
            return self.imputer.transform(samples[self.numeric_columns])
        self.imputer = IterativeImputer(
            max_iter=100, tol=1e-5, initial_strategy="median", imputation_order="roman"
        )
        self.train_df.loc[:, self.numeric_columns] = self.imputer.fit_transform(
            self.train_df[self.numeric_columns]
        )
        self.test_df.loc[:, self.numeric_columns] = self.imputer.transform(
            self.test_df[self.numeric_columns]
        )

    def _preprocess_mil_eng_pow(
        self, samples: pd.DataFrame = pd.DataFrame()
    ) -> Union[None, pd.DataFrame]:
        datasets = [self.train_df, self.test_df] if samples.empty else [samples]
        for dataset in datasets:
            if "mileage" in dataset.columns:
                dataset.loc[:, "mileage"] = (
                    dataset["mileage"]
                    .astype("str")
                    .str.extract(r"(\d+.\d+)")
                    .astype("float")[0]
                )
            if "engine" in dataset.columns:
                dataset.loc[:, "engine"] = (
                    dataset["engine"]
                    .astype("str")
                    .str.extract(r"(\d+)")[0]
                    .astype("float")
                )
            if "max_power" in dataset.columns:
                dataset.loc[:, "max_power"] = (
                    dataset["max_power"]
                    .astype("str")
                    .str.extract(r"([\d\.]+)")[0]
                    .astype("float")
                )
        if not samples.empty:
            return datasets[0]

    def _cast_to_int(
        self, samples: pd.DataFrame = pd.DataFrame()
    ) -> Union[None, pd.DataFrame]:
        datasets = [self.train_df, self.test_df] if samples.empty else [samples]
        for dataset in datasets:
            for col in ["engine", "seats", "max_torque_rpm"]:
                if col in dataset.columns:
                    dataset.loc[:, col] = dataset[col].astype("int")
        if not samples.empty:
            return datasets[0]

    def _add_model_from_name(
        self, samples: pd.DataFrame = pd.DataFrame()
    ) -> Union[None, pd.DataFrame]:
        datasets = [self.train_df, self.test_df] if samples.empty else [samples]
        for dataset in datasets:
            dataset.loc[:, "name"] = dataset["name"].astype("str").str.split(" ")
            dataset["model"] = dataset["name"].str[1]
            dataset.loc[:, "name"] = dataset["name"].str[0]

        if not samples.empty:
            return datasets[0]

    # удаление дубликатов, как ни странно, уменьшает качество модели,
    # поэтому данный шаг не будет использован
    def _remove_dublicates_train(self) -> None:
        self.train_df.drop_duplicates(
            subset=self.train_df.columns.difference(["selling_price"]),
            keep="first",
            inplace=True,
        )

    def _encode_categorical(
        self, samples: pd.DataFrame = pd.DataFrame()
    ) -> Union[tuple[np.array, np.array], np.array]:

        if not samples.empty:
            return self.cat_encoder.transform(samples[self.cat_columns])
        self.cat_encoder = OneHotEncoder(
            drop="first", sparse=False, handle_unknown="infrequent_if_exist"
        )
        with warnings.catch_warnings(category=UserWarning):
            warnings.simplefilter("ignore")
            X_train_cat = self.cat_encoder.fit_transform(self.X_train[self.cat_columns])
            X_test_cat = self.cat_encoder.transform(self.X_test[self.cat_columns])
        return X_train_cat, X_test_cat

    def _scale_features(
        self,
        X_train_cat: pd.DataFrame,
        X_test_cat: pd.DataFrame,
        samples: pd.DataFrame = pd.DataFrame(),
        samples_cat: np.array = np.array([]),
    ) -> Union[None, np.array]:
        if samples.empty:
            self.robust_scaler = RobustScaler()
            X_train = self.X_train.drop(columns=self.cat_columns)
            X_test = self.X_test.drop(columns=self.cat_columns)

            X_train = self.robust_scaler.fit_transform(X_train)
            X_test = self.robust_scaler.transform(X_test)

            self.X_train = np.concatenate([X_train, X_train_cat], axis=1)
            self.X_test = np.concatenate([X_test, X_test_cat], axis=1)
        else:
            samples_numeric = samples.drop(columns=self.cat_columns)
            samples_numeric = self.robust_scaler.transform(samples_numeric)
            samples = np.concatenate([samples_numeric, samples_cat], axis=1)
            return samples

    def preprocess_datasets(
        self, samples: pd.DataFrame = pd.DataFrame()
    ) -> Union[None, np.array]:
        if samples.empty:
            self._preprocess_mil_eng_pow()
            self._create_features_from_torque()
            self._fill_na()
            self._cast_to_int()
            self._add_model_from_name()

            self.X_train = self.train_df.drop(columns="selling_price")
            self.y_train = self.train_df["selling_price"]
            self.X_test = self.test_df.drop(columns="selling_price")
            self.y_test = self.test_df["selling_price"]

            X_train_cat, X_test_cat = self._encode_categorical()
            self._scale_features(X_train_cat, X_test_cat)
        else:
            samples = self._preprocess_mil_eng_pow(samples)
            samples = self._create_features_from_torque(samples)
            samples.loc[:, self.numeric_columns] = self._fill_na(samples)
            samples = self._cast_to_int(samples)
            samples = self._add_model_from_name(samples)

            if "selling_price" in samples.columns:
                samples = samples.drop(columns="selling_price")

            samples_cat = self._encode_categorical(samples)
            samples = self._scale_features(
                pd.DataFrame(), pd.DataFrame(), samples, samples_cat
            )
            return samples

    def predict_samples(self, samples: pd.DataFrame) -> np.array:
        samples = self.preprocess_datasets(samples)
        return np.exp(self.regressor.predict(samples))

    def _choose_best_model(self) -> dict:
        def objective(trial):
            alpha = trial.suggest_float(
                "alpha", self.alpha_start, self.alpha_end, log=True
            )
            random_state = trial.suggest_int(
                "random_state", self.random_state_start, self.random_state_end
            )
            solver = trial.suggest_categorical("solver", self.solvers)
            regressor = Ridge(alpha=alpha, solver=solver, random_state=random_state)
            regressor.fit(self.X_train, np.log(self.y_train))

            r_test_mse = mean_squared_error(
                self.y_test, np.exp(regressor.predict(self.X_test)), squared=False
            )

            return r_test_mse

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        study.optimize(objective, n_trials=self.n_trials, n_jobs=-1)
        return study.best_params

    def train_model(self) -> None:
        self.best_params = self._choose_best_model()

        self.regressor = Ridge(**self.best_params)
        self.regressor.fit(self.X_train, np.log(self.y_train))

    def _get_cv_scores(self) -> dict:
        self.cv_scores = {}
        splitter = ShuffleSplit(n_splits=self.n_splits, test_size=0.2)
        for i, (train_index, test_index) in enumerate(splitter.split(self.X_train)):
            X_train_for_cv, y_train_for_cv = (
                self.X_train[train_index, :],
                self.y_train[train_index],
            )
            X_test_for_cv, y_test_for_cv = (
                self.X_train[test_index, :],
                self.y_train[test_index],
            )

            regressor_for_split = Ridge(**self.best_params)
            regressor_for_split.fit(X_train_for_cv, np.log(y_train_for_cv))

            r2_score_cv = r2_score(
                y_test_for_cv, np.exp(regressor_for_split.predict(X_test_for_cv))
            )
            rmse_score_cv = mean_squared_error(
                y_test_for_cv,
                np.exp(regressor_for_split.predict(X_test_for_cv)),
                squared=False,
            )

            self.cv_scores[f"R2_score_{i}"] = r2_score_cv
            self.cv_scores[f"RMSE_score_{i}"] = rmse_score_cv

        r2_cv = 0
        rmse_cv = 0
        for i in range(self.n_splits):
            r2_cv += self.cv_scores[f"R2_score_{i}"]
            rmse_cv += self.cv_scores[f"RMSE_score_{i}"]

        self.cv_scores["R2_score_mean"] = r2_cv / self.n_splits
        self.cv_scores["RMSE_score_mean"] = rmse_cv / self.n_splits
        return self.cv_scores

    def evaluate_model(self) -> tuple[dict, dict]:
        r2_score_train = r2_score(
            self.y_train, np.exp(self.regressor.predict(self.X_train))
        )
        r2_score_test = r2_score(
            self.y_test, np.exp(self.regressor.predict(self.X_test))
        )

        rmse_score_train = mean_squared_error(
            self.y_train, np.exp(self.regressor.predict(self.X_train)), squared=False
        )
        rmse_score_test = mean_squared_error(
            self.y_test, np.exp(self.regressor.predict(self.X_test)), squared=False
        )

        self.scores = {
            "R2_score_train": r2_score_train,
            "R2_score_test": r2_score_test,
            "RMSE_score_train": rmse_score_train,
            "RMSE_score_test": rmse_score_test,
        }
        self._get_cv_scores()

        return self.scores, self.cv_scores

    def save_model(self, add_save_path: str = "") -> None:
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.saved = True
        self.save_folder = os.path.join(
            self.save_path, datetime.now().strftime("%d-%m-%Y %H-%M")
        )
        if add_save_path:
            self.save_folder = add_save_path
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        self.model_path = os.path.join(self.save_folder, "model.pkl")
        self.imputer_path = os.path.join(self.save_folder, "imputer.pkl")
        self.encoder_path = os.path.join(self.save_folder, "encoder.pkl")
        self.scaler_path = os.path.join(self.save_folder, "scaler.pkl")

        with open(self.model_path, "wb") as model_file:
            pickle.dump(self.regressor, model_file)
        with open(self.imputer_path, "wb") as imputer_file:
            pickle.dump(self.imputer, imputer_file)
        with open(self.encoder_path, "wb") as encoder_file:
            pickle.dump(self.cat_encoder, encoder_file)
        with open(self.scaler_path, "wb") as scaler_file:
            pickle.dump(self.robust_scaler, scaler_file)

    def load_model(self, path_to_saved_model: str = "") -> None:
        if self.saved or path_to_saved_model:

            if path_to_saved_model:
                self.save_folder = path_to_saved_model
                self.model_path = os.path.join(self.save_folder, "model.pkl")
                self.imputer_path = os.path.join(self.save_folder, "imputer.pkl")
                self.encoder_path = os.path.join(self.save_folder, "encoder.pkl")
                self.scaler_path = os.path.join(self.save_folder, "scaler.pkl")

            with open(self.model_path, "rb") as model_file:
                self.regressor = pickle.load(model_file)
            with open(self.imputer_path, "rb") as imputer_file:
                self.imputer = pickle.load(imputer_file)
            with open(self.encoder_path, "rb") as encoder_file:
                self.cat_encoder = pickle.load(encoder_file)
            with open(self.scaler_path, "rb") as scaler_file:
                self.robust_scaler = pickle.load(scaler_file)
