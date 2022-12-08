import lightning as L
from lightning.app.storage import Path
from lightning.app.structures import List
import numpy as np
import pickle
from sklearn import datasets
import sklearn.metrics as skm
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
import time


class RootComponent(L.LightningFlow):
    def __init__(self, n_splits):
        super().__init__()
        self.n_splits = n_splits
        self.split_paths = [Path(f"fold_{fold}_split.pickle") for fold in range(n_splits)]
        self.test_path = Path("test.pickle")
        self.model_paths = [Path(f"fold_{fold}_model.pickle") for fold in range(n_splits)]
        self.split_data_work = SplitDataWork()
        self.train_works = List()
        self.splitted = False
        self.trained = False
        self.predicted = False
        self.predict_work = PredictWork()

    def run(self) -> None:
        if not self.splitted:
            self.split_data_work.run(self.split_paths, self.test_path)
            self.splitted = True
        if self.splitted and (not self.trained):
            for split_path, model_path in zip(self.split_paths, self.model_paths):
                self.train_works.append(TrainWork(parallel=True))
                self.train_works[-1].run(split_path, model_path)
            self.trained = True
        if self.splitted and self.trained and (not self.predicted):
            self.predict_work.run(self.test_path, self.model_paths)
            self.predicted = True


class SplitDataWork(L.LightningWork):
    def run(self, split_paths, test_path):
        data = datasets.load_breast_cancer()
        X, y = data.data, data.target
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
        with open(test_path, "wb") as outf:
            pickle.dump((X_test, y_test), outf)
        kf = KFold(n_splits=len(split_paths))
        for split_path, (train_idx, val_idx) in zip(split_paths, kf.split(X_train_val)):
            X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
            X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]
            with open(split_path, "wb") as outf:
                pickle.dump((X_train, y_train, X_val, y_val), outf)


class TrainWork(L.LightningWork):
    def run(self, split_path, model_path):
        print("split_path:", split_path)
        print("model_path:", model_path)
        with open(split_path, "rb") as inf:
            X_train, y_train, X_test, y_test = pickle.load(inf)
        fold = split_path.stem.split("_")[1]
        tree = RandomForestClassifier()
        tree.fit(X_train, y_train)
        with open(model_path, "wb") as outf:
            pickle.dump(tree, outf)
    

class PredictWork(L.LightningWork):
    def run(self, test_path, model_paths):
        time.sleep(5)
        with open(test_path, "rb") as inf:
            X_test, y_test = pickle.load(inf)
        y_prob = []
        for path in model_paths:
            with open(path, "rb") as inf:
                model = pickle.load(inf)
            temp = model.predict_proba(X_test)[:, 1]
            print("Average precision (single) :", skm.average_precision_score(y_test, temp))
            y_prob.append(temp)
        y_prob = np.mean(y_prob, axis=0)
        print("Average precision (ensemble):", skm.average_precision_score(y_test, y_prob))


app = L.LightningApp(RootComponent(4))
