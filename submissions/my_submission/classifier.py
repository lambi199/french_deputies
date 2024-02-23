import optuna
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import optuna.visualization as vis

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
 

class Classifier(BaseEstimator):
    def __init__(self):
        self.pipe = Pipeline(
            steps=[
                ("model", HistGradientBoostingClassifier(random_state=1234)),
            ]
        )
        

    def objective(self, trial, X, y):

        learning_rate = trial.suggest_float(name="model__learning_rate", low=0.01, high=0.1, log=True)
        max_iter = trial.suggest_int(name="model__max_iter", low=100, high=500)
        max_depth = trial.suggest_int(name="model__max_depth", low=3, high=20, step=1)
        min_samples_leaf = trial.suggest_int(name="model__min_samples_leaf", low=1, high=5, step=1)
        max_bins = trial.suggest_int(name="model__max_bins", low=2, high=255)

        self.pipe.set_params(
            model__learning_rate=learning_rate,
            model__max_iter=max_iter,
            model__max_depth=max_depth,
            model__min_samples_leaf=min_samples_leaf,
            model__max_bins=max_bins,
        )

        self.pipe.fit(X, y)
        cv_score = cross_val_score(self.pipe, X, y, n_jobs=4, cv=5)
        mean_cv_accuracy = cv_score.mean()
        return mean_cv_accuracy

    def fit(self, X, y):
        study = optuna.create_study(direction='maximize')  
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=10)
        vis.plot_optimization_history(study)
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
        self.pipe.set_params(**best_params)
        self.pipe.fit(X, y)
  
    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)
    
    def score(self, X, y):
        return self.pipe.score(X, y)