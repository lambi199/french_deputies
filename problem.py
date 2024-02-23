import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
import rampwf as rw
import numpy as np

problem_title = "Political party prediction from deputies data"

columns_to_keep = ['civ',
                   'age',
                   'experienceDepute',
                   'scoreParticipation',
                   'scoreParticipationSpecialite',
                   'scoreLoyaute',
                   'scoreMajorite']

int_to_cat = {
    0: "SOC-A",
    1: "LR",
    2: "RE",
    3: "LIOT",
    4: "HOR",
    5: "DEM",
    6: "LFI-NUPES",
    7: "GDR-NUPES",
    8: "RN",
    9: "ECOLO",
}

cat_to_int = {v: k for k, v in int_to_cat.items()}

_parties_label_int = list(int_to_cat)

Predictions = rw.prediction_types.make_multiclass(label_names=_parties_label_int)
workflow = rw.workflows.Classifier()


score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3),
]

def _get_data(path = ".",split ="train"):
    data_df = pd.read_csv(os.path.join(path, "data","public", split + ".csv"))
    X = data_df.loc[:, columns_to_keep]
    y = np.array(data_df["groupeAbrev"].map(cat_to_int).fillna(-1).astype("int8"))

    return X,y

def get_train_data(path="."):
    data = pd.read_csv(os.path.join(path, "data", "public", "train" + ".csv"))
    data_df = data.copy()
    SampleID = data_df["id"]
    global groups
    groups = SampleID
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")

def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2)
    return cv.split(X, y, groups)




