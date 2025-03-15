import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def load_housing_data(csv_path):
    """Lädt die Housing-Daten aus einer CSV-Datei."""
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio=0.2):
    """Teilt die Daten in Trainings- und Test-Sets."""
    return train_test_split(data, test_size=test_ratio, random_state=42)


def prepare_data(housing):
    """Bereitet die Housing-Daten für das Training vor."""
    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared, full_pipeline
