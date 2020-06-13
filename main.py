#from operators.snowflake.warehouse import SnowflakeOperator
from operators.gcp.bigquery import BigQueryOperator
from metaflow import FlowSpec, step, Parameter
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error



os.environ["USERNAME"] = "soy_un_usuario"  # needed for metaflow


class Pipeline(FlowSpec):
    un_parametro = Parameter('un_parametro',
                                  help='Un ejemplo de parámetro',
                                  default=3)

    @step
    def start(self):
        self.next(self.read_and_process_data)

    @step
    def read_and_process_data(self):
        """
        Reads data from the web and prepares it for training
        """
        df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                         sep=';')
        bins = (2, 6, 8)

        group_names = ['bad', 'good']
        df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)
        label_quality = preprocessing.LabelEncoder()
        # Bad becomes 0 and good becomes 1
        df['quality'] = label_quality.fit_transform(df['quality'])
        df['quality'].value_counts()

        X = df.drop('quality', axis=1)
        y = df['quality']

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=50)

        self.y_train = y_train
        self.y_test = y_test

        sc = preprocessing.StandardScaler()
        self.X_train = sc.fit_transform(X_train)
        self.X_test = sc.transform(X_test)

        self.next(self.train_model1, self.train_model2)
    

    @step
    def train_model1(self):
        """
        Train model version 1
        """
        model = SGDClassifier()
        model_name = "SGD Classifier"
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(
            model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
        msg = "%s: %f (%f)" % (model_name, cv_results.mean(), cv_results.std())
        print(msg)

        self.accuracy = cv_results.mean()
        self.model_name = model_name
        self.next(self.get_best_model)


    @step
    def train_model2(self):
        """
        Train model version 2
        """
        model = RandomForestClassifier(n_estimators=500)
        model_name = "Random Forest Classifier"
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(
            model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
        msg = "%s: %f (%f)" % (model_name, cv_results.mean(), cv_results.std())
        print(msg)

        self.accuracy = cv_results.mean()
        self.model_name = model_name
        self.next(self.get_best_model)


    @step
    def get_best_model(self, inputs):
        """
        Select the best model
        """
        accuracies = np.array([inputs.train_model1.accuracy,
                              inputs.train_model2.accuracy])
        names = [inputs.train_model1.model_name, inputs.train_model2.model_name]
        best_acc_ind = np.argmax(accuracies)
        print("Mejor modelo: ", names[best_acc_ind])
        print("Mejor accuracy: ", accuracies[best_acc_ind])
        self.next(self.end)


    @step
    def end(self):
        print("Pipeline done!")


if __name__ == '__main__':
    Pipeline()
