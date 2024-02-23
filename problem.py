import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
import rampwf as rw
import numpy as np

problem_title = "Party prediction from loyalty data"


remove_deputies_with_no_party ='NI'

columns_to_keep = ['id',
                   'civ',
                   'age',
                   'experienceDepute',
                   'groupeAbrev', 
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

# Mapping categories to int
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

def convert_to_days(value):

    if 'nan' in str(value):
        return value
    
    number, time = value.split(' ')
    
    if time == 'mois':
        return int(number) * 30.4167
    elif time == 'ans' or time == 'an':
        return int(number) * 365  
    else:
        return int(number)
    
def convert_gender(value):
    if str(value).strip() == "Mme":
        return 0
    else:
        return 1


def _get_data(path = ".",split ="train"):
    data_df = pd.read_csv(os.path.join(path, "data", split + "_deputes-actives" + ".csv"))
    data_df['experienceDepute'] = data_df['experienceDepute'].apply(convert_to_days)
    data_df['civ'] = data_df['civ'].apply(convert_gender)
    
    data_df = data_df.loc[:, columns_to_keep]
    remove_data = data_df[(data_df['groupeAbrev'] == remove_deputies_with_no_party) | (data_df['groupeAbrev'] == '2024-02-20')].index
    cleaned_data_df = data_df.drop(remove_data)


 

    data_df["groupeAbrev"] = data_df["groupeAbrev"].astype("category")

    X_data = ['civ',
              'age',
              'experienceDepute',
              'scoreParticipation',
              'scoreParticipationSpecialite',
              'scoreLoyaute',
              'scoreMajorite']
        
    X = cleaned_data_df.loc[:, X_data]
    y = np.array(cleaned_data_df["groupeAbrev"].map(cat_to_int).fillna(-1).astype("int8"))

    return X,y

def get_train_data(path="."):
    data = pd.read_csv(os.path.join(path, "data", "train" + "_deputes-actives" + ".csv"))
    data_df = data.copy()
    remove_data = data_df[(data_df['groupeAbrev'] == remove_deputies_with_no_party) | (data_df['groupeAbrev'] == '2024-02-20')].index
    cleaned_data_df = data_df.drop(remove_data)
    SampleID = cleaned_data_df["id"]
    global groups
    groups = SampleID
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")

def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2)
    return cv.split(X, y, groups)




