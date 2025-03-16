import os
from MaLe.Unterrichtsblock_1.Data_Preparation import load_housing_data, split_train_test, prepare_data
from MaLe.Unterrichtsblock_1.Model_Training import train_model, evaluate_model


# Dateipfad der CSV-Datei
HOUSING_PATH = os.path.join("datasets", "housing", "housing.csv")

# Daten laden
housing = load_housing_data(HOUSING_PATH)

# Trainings- und Test-Sets erstellen
train_set, test_set = split_train_test(housing)

# Trainingsdaten vorbereiten
housing_prepared, pipeline = prepare_data(train_set)
y_train = train_set["median_house_value"].values

# Modell trainieren
model = train_model(housing_prepared, y_train)

# Testdaten vorbereiten
X_test_prepared = pipeline.transform(test_set)
y_test = test_set["median_house_value"].values

# Modell bewerten
evaluate_model(model, X_test_prepared, y_test)
