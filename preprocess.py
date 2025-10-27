import pandas as pd
import numpy as np
import sklearn 
from sklearn.base import BaseEstimator, TransformerMixin
import sys


class CustomPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, handle_outliers=True, firstN=True, id_cols=None, rare_cols=None):
        self.handle_outliers = handle_outliers
        self.firstN = firstN
        self.id_cols = id_cols or [
            'id', 'url', 'image_url', 'region_url', 'VIN',
            'description', 'cylinders', 'posting_date', 'posting_year'
        ]
        self.rare_cols = rare_cols or ['region', 'model']

        # параметры обработки 
        self.__values_for_num_cols = None
        self.__values_for_cot_cols = None
        self.__data_fill_time_values = {}
        self.__cols_to_drop = None
        self.__top100_region = None
        self.__top250_model = None

        # Метка обученности
        self.fitted = False


    def fit(self, X: pd.DataFrame, y=None):
        """Обучение препроцессора: вычисление и запоминание всех статистик."""
     
        # Преобразуем цилиндры
        X['cylinders_num'] = X['cylinders'].str.extract('(\d+)').astype(float)

        # Вычисляем статистики
        num_cols = self.__find_num_col_to_fill(X)
        self.__values_for_num_cols = X[num_cols].median(numeric_only=True)
        self.__values_for_cot_cols = 'unknown'

        # Обработка временных признаков
        X['posting_date'] = pd.to_datetime(X['posting_date'], errors='coerce', utc=True)
        X['posting_year'] = X['posting_date'].dt.year
        X['posting_month'] = X['posting_date'].dt.month
        X['posting_day'] = X['posting_date'].dt.day
        X['posting_weekday'] = X['posting_date'].dt.weekday
        X['posting_hour'] = X['posting_date'].dt.hour

        # Запоминаем моду временных признаков
        for feature in ['posting_year', 'posting_month', 'posting_day', 'posting_weekday', 'posting_hour']:
            if feature in X.columns:
                mode_value = X[feature].mode()
                self.__data_fill_time_values[feature] = mode_value.iloc[0] if not mode_value.empty else 0

        # Запоминаем колонки с большим количеством пропусков
        self.__cols_to_drop = self.__find_col_to_del(X)

        # Запоминаем топ-значения
        if self.firstN:
            self.__top100_region = X['region'].value_counts().head(100).index
            self.__top250_model = X['model'].value_counts().head(250).index

        self.fitted = True
        return self


    def transform(self, X: pd.DataFrame):
        """Применяет запомненные преобразования (без удаления выбросов и строк)."""
        if not self.fitted:
            raise ValueError("Сначала вызовите fit() на обучающих данных.")

        X = X.copy()

        # Удаляем только колонки с большим количеством пропусков (по статистике train)
        X.drop(columns=self.__cols_to_drop, errors='ignore', inplace=True)

        # Преобразуем цилиндры
        X['cylinders_num'] = X['cylinders'].str.extract('(\d+)').astype(float)

        # Создание новых временных признаков
        X['posting_date'] = pd.to_datetime(X['posting_date'], errors='coerce', utc=True)
        X['posting_year'] = X['posting_date'].dt.year
        X['posting_month'] = X['posting_date'].dt.month
        X['posting_day'] = X['posting_date'].dt.day
        X['posting_weekday'] = X['posting_date'].dt.weekday
        X['posting_hour'] = X['posting_date'].dt.hour

        # Заполнение пропусков во временных признаках
        for feature, value in self.__data_fill_time_values.items():
            if feature in X.columns:
                X[feature] = X[feature].fillna(value)

        # Признак длины описания
        X['description'] = X['description'].fillna('')
        X['len_description'] = X['description'].str.len()

        # Удаляем идентификаторы
        X.drop(columns=self.id_cols, errors='ignore', inplace=True)

        # Заполняем числовые пропуски
        num_cols = self.__find_num_col_to_fill(X)
        if num_cols:
            X[num_cols] = X[num_cols].fillna(self.__values_for_num_cols)

        # Заполняем категориальные пропуски
        cot_cols = self.__find_cot_col_to_fill(X)
        if cot_cols:
            X[cot_cols] = X[cot_cols].fillna(self.__values_for_cot_cols)

        # Обработка редких категорий
        if self.firstN:
            if 'region' in X.columns:
                X['region_top100'] = X['region'].apply(lambda x: x if x in self.__top100_region else 'Other')
                X.drop(columns=['region'], inplace=True, errors='ignore')
            if 'model' in X.columns:
                X['model_top250'] = X['model'].apply(lambda x: x if x in self.__top250_model else 'Other')
                X.drop(columns=['model'], inplace=True, errors='ignore')

        return X

    def __find_num_col_to_fill(self, data: pd.DataFrame):
        num_cols = data.select_dtypes(exclude=['object'])
        return num_cols.columns[num_cols.isnull().any()].tolist()

    def __find_cot_col_to_fill(self, data: pd.DataFrame):
        cot_cols = data.select_dtypes(include=['object'])
        return [col for col in cot_cols.columns if cot_cols[col].isnull().any() and col != 'description']

    def __find_col_to_del(self, data: pd.DataFrame):
        all_k = data.shape[0]
        cols_to_drop = [col for col in data.columns if data[col].isnull().mean() > 0.7]
        return cols_to_drop



