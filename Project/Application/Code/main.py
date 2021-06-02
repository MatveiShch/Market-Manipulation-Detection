import sys
import os
from PyQt5 import QtWidgets
import design
import pathlib
import yfinance as yf
from datetime import timedelta
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.ensemble import GradientBoostingClassifier

days_for_normalization = 20


class NCAClassifier:
    def __init__(self, par):
        self.knn = KNeighborsClassifier(n_neighbors=par[1], weights='distance')
        self.nca = NeighborhoodComponentsAnalysis(n_components=par[0], max_iter=10000000, tol=0.0)

    def fit(self, x_train, y_train):
        self.nca.fit(x_train, y_train)
        print(self.nca.n_iter_)
        print(self.nca.components_)
        x_train = self.nca.transform(x_train)
        self.knn.fit(x_train, y_train)
        return self

    def reduce(self, x):
        return self.nca.transform(x)

    def predict(self, x_test):
        x_test = self.nca.transform(x_test)
        return self.knn.predict(x_test)


def standartization(comp):
    cols = ['Volume', 'Volume variance',

            'Open with volume', 'Open with volume variance', 'High with volume', 'High with volume variance',
            'Low with volume',
            'Low with volume variance', 'Close with volume', 'Close with volume variance',

            'Open minus close with volume', 'Open minus close with volume variance', 'High minus close with volume',
            'High minus close with volume variance', 'Low minus close with volume',
            'Low minus close with volume variance',

            'Volume percentage change', 'Volume percentage change variance',

            'Open percentage change', 'Open percentage change variance', 'High percentage change',
            'High percentage change variance', 'Low percentage change', 'Low percentage change variance',
            'Close percentage change', 'Close percentage change variance']

    newcomp = pd.DataFrame(columns=cols)

    for j in comp.index:
        if j > days_for_normalization - 1:
            d = j - days_for_normalization

            row = []

            volumes = comp.iloc[d:j + 1, 5].reset_index(drop=True)
            sum_of_volumes = volumes.sum(axis=0)
            average_volume = sum_of_volumes / (days_for_normalization + 1)

            volumes_minus_average = volumes - average_volume
            volumes_minus_average_squared = volumes_minus_average ** 2
            sum_of_volumes_minus_average_squared = volumes_minus_average_squared.sum(axis=0)
            sample_variance = sum_of_volumes_minus_average_squared / days_for_normalization

            if sample_variance == 0:
                row.append(0.0)
            else:
                row.append((comp.iloc[j, 5] - average_volume) / (4 * math.sqrt(sample_variance)))
            row.append(sample_variance)

            sets = []

            for l in [0, 1, 2, 3]:
                sets.append(comp.iloc[d:j + 1, l].reset_index(drop=True))
            for l in [0, 1, 2]:
                sets.append((comp.iloc[d:j + 1, l] - comp.iloc[d:j + 1, 3]).reset_index(drop=True))
            for i in sets:
                prices = i
                prices_by_volumes = prices * volumes
                sum_of_prices_by_volume = prices_by_volumes.sum(axis=0)
                average_price = sum_of_prices_by_volume / sum_of_volumes

                prices_minus_average = prices - average_price
                prices_minus_average_squared = prices_minus_average ** 2
                prices_minus_average_squared_by_volumes = prices_minus_average_squared * volumes
                sum_of_prices_minus_average_squared_by_volumes = prices_minus_average_squared_by_volumes.sum(axis=0)
                sample_variance = sum_of_prices_minus_average_squared_by_volumes / sum_of_volumes

                if sample_variance == 0:
                    row.append(0.0)
                else:
                    row.append((i.iloc[days_for_normalization] - average_price) / (4 * math.sqrt(sample_variance)))
                row.append(sample_variance)

            sets = []

            for l in [0, 1, 2, 3, 5]:
                sets.append(comp.iloc[d:j + 1, l].reset_index(drop=True))

            for i in sets:
                prices = i

                prev_prices = prices.iloc[0:days_for_normalization].reset_index(drop=True)
                new_prices = prices.iloc[1:days_for_normalization + 1].reset_index(drop=True)

                prices = (new_prices - prev_prices) * 100 / prev_prices

                sum_of_prices = prices.sum(axis=0)
                average_price = sum_of_prices / days_for_normalization

                prices_minus_average = prices - average_price
                prices_minus_average_squared = prices_minus_average ** 2

                sum_of_prices_minus_average_squared = prices_minus_average_squared.sum(axis=0)

                sample_variance = sum_of_prices_minus_average_squared / (days_for_normalization - 1)

                if sample_variance == 0:
                    row.append(0.0)
                else:
                    row.append(
                        (prices.iloc[days_for_normalization - 1] - average_price) / (4 * math.sqrt(sample_variance)))
                row.append(sample_variance)

            newcomp.loc[len(newcomp)] = row
    return newcomp


# this method collects the trading data from the Yahoo! Finance and detects market manipulation
# returns pandas DataFrame with dates and results from each model
def detectionResults(symbol, startDate, endDate, to_include_KNN, to_include_GBT, path):
    data = pd.DataFrame()
    try:
        data = yf.download(symbol, startDate - timedelta(days=days_for_normalization * 7), endDate + timedelta(days=1))
    except:
        return 'The Ticker Symbol or the chosen trade period is not present in the Yahoo! Finance!'

    try:
        indeces_to_delete = data[data['Volume'] == 0].index
        data.drop(indeces_to_delete, inplace=True)

        if data.isnull().any().any() or (np.inf in data.values) or (-np.inf in data.values) or (0.0 in data.values):
            return 'The Yahoo! Finance dataset contains inappropriate values!'
    except:
        return 'The Yahoo! Finance dataset contains inappropriate values!'
    print(data)

    try:
        days_before_startDate = len(data.loc[:startDate - timedelta(days=1)])
        if days_before_startDate > days_for_normalization:
            data.drop(data.index[range(0, days_before_startDate - days_for_normalization)], inplace=True)
    except:
        return 'Failed eliminate the irrelevant dates'

    if len(data) <= days_for_normalization:
        return 'The chosen period has no trading days'
    data_without_dates = pd.DataFrame()

    try:
        data_without_dates = data.reset_index(drop=True)
    except:
        return 'Failed to reset index'

    standarised_data = pd.DataFrame()

    try:
        standarised_data = standartization(data_without_dates)
    except:
        return 'Failed to standarise data'

    data.reset_index(inplace=True)
    results = data[days_for_normalization:].drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)
    results.reset_index(drop=True, inplace=True)

    try:
        if to_include_KNN == True:
            KNN = pickle.load(open('KNNModel.sav', 'rb'))
            KNN_results = pd.DataFrame(KNN.predict(standarised_data), columns=['KNN'])
            results = pd.concat([results, KNN_results], axis=1)
    except:
        return 'Failed to load KNN results'

    try:
        if to_include_GBT == True:
            GBT = pickle.load(open('GBTModel.sav', 'rb'))
            GBT_results = pd.DataFrame(GBT.predict(standarised_data), columns=['GBT'])
            results = pd.concat([results, GBT_results], axis=1)
    except:
        return 'Failed to load GBT results'

    full_path = path + "/Market_Manipulation_Analysis_" + symbol + ".xlsx"
    results.to_excel(full_path, index=False)

    errorLine = 'No error detected'
    return errorLine


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        desktop = pathlib.Path.home()
        self.OutputPathLine.setText(str(desktop))

        self.DetectButton.clicked.connect(self.detect)
        self.ChooseButton.clicked.connect(self.choose)

    def detect(self):
        errorLine = ''
        self.ErrorLine.clear()

        to_include_KNN = bool(self.KNNBox.checkState() // 2)
        to_include_GBT = bool(self.GBTBox.checkState() // 2)

        if to_include_KNN == False and to_include_GBT == False:
            errorLine = 'There is no selected model!'
        else:
            symbol = self.TickerSymbolSpace.text()
            if symbol == '':
                errorLine = 'The is no entered Ticker Symbol!'
            else:
                start = self.StartDateSpace.date().toPyDate()
                end = self.EndDateSpace.date().toPyDate()
                if start > end:
                    errorLine = 'The start date is greater than the end date!'
                else:
                    errorLine = detectionResults(symbol, start, end, to_include_KNN, to_include_GBT,
                                                 self.OutputPathLine.text())

        self.ErrorLine.setText(errorLine)

    def choose(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose the output path",
                                                               self.OutputPathLine.text())

        if directory:
            self.OutputPathLine.setText(directory)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
