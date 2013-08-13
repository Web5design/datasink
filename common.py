"""
    datasink: A Pipeline for Large-Scale Heterogeneous Ensemble Learning
    Copyright (C) 2013 Sean Whalen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
"""

from numpy import sqrt, where
from pandas import DataFrame, concat, read_csv
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from scipy.io.arff import loadarff

def get_best_performer(df, one_se = False):
    if not one_se:
        return df[df.auc == df.auc.max()].head(1)
    se = df.auc.std() / sqrt(df.shape[0] - 1)
    return df[df.auc >= df.auc.max() - se].head(1)


def eval_performance(labels, predictions, false_discovery_rate = 0.1):
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    max_fpr_index = where(fpr >= false_discovery_rate)[0][0]
    print 'true positive rate: %.2f threshold: %.2f auc: %.3f' % (tpr[max_fpr_index], thresholds[max_fpr_index], roc_auc_score(labels, predictions))
    print confusion_matrix(labels, predictions > thresholds[max_fpr_index])


def load_arff(filename):
    return DataFrame.from_records(loadarff(filename)[0])


def load_properties(dirname):
    properties = [_.split('=') for _ in open(dirname + '/weka.properties').readlines()]
    d = {}
    for key, value in properties:
        d[key.strip()] = value.strip()
    return d


def read_fold(path, fold):
    train_df        = read_csv('%s/validation-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
    test_df         = read_csv('%s/predictions-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
    train_labels    = train_df.index.get_level_values('label').values
    test_labels     = test_df.index.get_level_values('label').values
    return train_df, train_labels, test_df, test_labels


def unbag(df, bag_count):
    cols = []
    bag_start_indices = range(0, df.shape[1], bag_count)
    names = [_.split('.')[0] for _ in df.columns.values[bag_start_indices]]
    for i in bag_start_indices:
        cols.append(df.ix[:, i:i+bag_count].mean(axis = 1))
    df = concat(cols, axis = 1)
    df.columns = names
    return df
