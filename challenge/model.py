import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def _get_period_day(self, date_str: str) -> str:
        # This function should return the period of the day of the flight.
        date_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()
        
        if morning_min <= date_time <= morning_max:
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        elif (evening_min <= date_time <= evening_max) or (night_min <= date_time <= night_max):
            return 'noche'
        return 'noche'

    def _is_high_season(self, fecha_str: str) -> int:
        # This function should return 1 if the flight is in high season, 0 otherwise.
        fecha_año = int(fecha_str.split('-')[0])
        fecha = datetime.strptime(fecha_str, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)
        
        if ((range1_min <= fecha <= range1_max) or 
            (range2_min <= fecha <= range2_max) or 
            (range3_min <= fecha <= range3_max) or
            (range4_min <= fecha <= range4_max)):
            return 1
        return 0

    def _get_min_diff(self, fecha_o: str, fecha_i: str) -> float:
        # This function should return the difference in minutes between the actual operation time and the scheduled time.
        fecha_o_dt = datetime.strptime(fecha_o, '%Y-%m-%d %H:%M:%S')
        fecha_i_dt = datetime.strptime(fecha_i, '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o_dt - fecha_i_dt).total_seconds()) / 60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data = data.copy()
        
        if 'Fecha-I' not in data.columns:
            raise ValueError("Missing required column: Fecha-I")
        if 'OPERA' not in data.columns:
            raise ValueError("Missing required column: OPERA")
        if 'TIPOVUELO' not in data.columns:
            raise ValueError("Missing required column: TIPOVUELO")
        if 'MES' not in data.columns:
            raise ValueError("Missing required column: MES")

        if target_column == "delay":
            if 'delay' not in data.columns:
                if 'Fecha-O' not in data.columns:
                    raise ValueError("Cannot compute delay: missing Fecha-O column")
                data['min_diff'] = data.apply(
                    lambda row: self._get_min_diff(row['Fecha-O'], row['Fecha-I']), 
                    axis=1
                )
                threshold_in_minutes = 15
                data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        opera_dummies = pd.get_dummies(data['OPERA'], prefix='OPERA')
        tipovuelo_dummies = pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO')
        mes_dummies = pd.get_dummies(data['MES'], prefix='MES')
        
        features = pd.concat([opera_dummies, tipovuelo_dummies, mes_dummies], axis=1)

        TOP_FEATURES = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

        for feature in TOP_FEATURES:
            if feature not in features.columns:
                features[feature] = 0

        features = features[TOP_FEATURES]

        if target_column is not None:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            target = pd.DataFrame(data[target_column])
            return features, target
        
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Fist make validations for the input data
        if features.shape[0] != target.shape[0]:
            raise ValueError("Features and target must have the same number of rows")
        
        if target.shape[1] != 1:
            raise ValueError("Target must be a single column DataFrame")
        
        target_series = target.iloc[:, 0]
        
        # Calculate the number of positive and negative samples
        n_y0 = len(target_series[target_series == 0])
        n_y1 = len(target_series[target_series == 1])
        
        if n_y1 == 0:
            raise ValueError("No positive samples in target")
        
        class_weight = {1: n_y0 / len(target_series), 0: n_y1 / len(target_series)}
        
        self._model = LogisticRegression(class_weight=class_weight, random_state=42, max_iter=1000)
        self._model.fit(features, target_series)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return [0] * len(features)
        
        predictions = self._model.predict(features)
        return [int(pred) for pred in predictions]