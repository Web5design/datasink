#!/usr/bin/env python

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

from os import mkdir
from os.path import abspath, exists
from sys import argv

from common import get_best_performer, load_properties, read_fold, unbag
from diversity import average_diversity_score
from numpy import column_stack
from numpy.random import seed
from pandas import DataFrame, concat
from sklearn.cluster import MiniBatchKMeans # https://github.com/scikit-learn/scikit-learn/issues/636
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score

def eval_metrics(df, labels, predictions, indices):
    auc         = roc_auc_score(labels, predictions)
    brier       = mean_squared_error(labels, predictions)
    diversity   = average_diversity_score(df.values)
    return DataFrame({'auc': auc, 'brier': brier, 'diversity': diversity}, index = indices)


def eval_cluster_metrics(x, labels, predictions, n_clusters, indices):
    auc         = roc_auc_score(labels, predictions)
    brier       = mean_squared_error(labels, predictions)
    diversity   = average_diversity_score(x)
    return DataFrame({'auc': auc, 'brier': brier, 'diversity': diversity, 'n_clusters': n_clusters}, index = indices)


def stack_intra(n_clusters, distances, fit_df, fit_labels, predict_df):
    cluster_labels = MiniBatchKMeans(n_clusters).fit_predict(distances)
    cols = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        model = stacker.fit(fit_df.ix[:, mask], fit_labels)
        predictions = model.predict_proba(predict_df.ix[:, mask])[:, 1]
        cols.append(predictions)
    values = column_stack(cols)
    predictions = values.mean(axis = 1)
    return values, predictions


def stack_inter(n_clusters, distances, fit_df, fit_labels, predict_df):
    cluster_labels = MiniBatchKMeans(n_clusters).fit_predict(distances)
    cols = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        predictions = fit_df.ix[:, mask].mean(axis = 1)
        cols.append(predictions)
    model = stacker.fit(column_stack(cols), fit_labels)
    cols = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        predictions = predict_df.ix[:, mask].mean(axis = 1)
        cols.append(predictions)
    values = column_stack(cols)
    predictions = model.predict_proba(values)[:, 1]
    return values, predictions


def stacked_selection(fold):
    seed(seedval)
    indices = [[fold], [seedval]]
    train_df, train_labels, test_df, test_labels = read_fold(path, fold)
    train_distances = 1 - train_df.corr().abs()
    train_metrics = []
    test_metrics = []
    for n_clusters in range(1, max_clusters + 1):
        train_values, train_predictions = stack_function(n_clusters, train_distances, train_df, train_labels, train_df)
        test_values, test_predictions = stack_function(n_clusters, train_distances, train_df, train_labels, test_df)
        train_metrics.append(eval_cluster_metrics(train_values, train_labels, train_predictions, n_clusters, indices))
        test_metrics.append(eval_cluster_metrics(test_values, test_labels, test_predictions, n_clusters, indices))
    best_cluster_size = get_best_performer(concat(train_metrics)).n_clusters
    test_values, test_predictions = stack_function(best_cluster_size, train_distances, train_df, train_labels, test_df)
    return eval_cluster_metrics(test_values, test_labels, test_predictions, best_cluster_size, indices), concat(test_metrics)


def stacked_generalization(fold):
    seed(seedval)
    train_df, train_labels, test_df, test_labels = read_fold(path, fold)
    if method == 'aggregate':
        train_df = unbag(train_df, bag_count)
        test_df = unbag(test_df, bag_count)
    predictions = stacker.fit(train_df, train_labels).predict_proba(test_df)[:, 1]
    return eval_metrics(test_df, test_labels, predictions, [[fold], [seedval]])


iterative_methods = set(['inter', 'intra'])
noniterative_methods = set(['aggregate', 'standard'])
methods = iterative_methods.union(noniterative_methods)

path = abspath(argv[1])
assert exists(path)
if not exists('%s/analysis' % path):
    mkdir('%s/analysis' % path)
method = argv[2]
assert method in methods
p = load_properties(path)
fold_count = int(p['foldCount'])
bag_count = int(p['bagCount'])
max_clusters = 20

# use shallow non-linear stacker by default
stacker = RandomForestClassifier(n_estimators = 200, max_depth = 2, bootstrap = False, random_state = 0)
if len(argv) > 3 and argv[3] == 'linear':
    stacker = SGDClassifier(loss = 'log', n_iter = 50, random_state = 0)
print stacker

index_labels = ['fold', 'seed']
if method in iterative_methods:
    stack_function = eval('stack_' + method)
    best_dfs = []
    iteration_dfs = []
    for seedval in range(10):
        results = Parallel(n_jobs = -1, verbose = 50)(delayed(stacked_selection)(fold) for fold in range(fold_count))
        for best_df, iteration_df in results:
            best_dfs.append(best_df)
            iteration_dfs.append(iteration_df)
    iteration_df = concat(iteration_dfs)
    iteration_df.to_csv('%s/analysis/stacking-%s-iterations.csv' % (path, method), index_label = index_labels)
else:
    seedval = 0
    best_dfs = Parallel(n_jobs = -1, verbose = 0)(delayed(stacked_generalization)(fold) for fold in range(fold_count))
best_df = concat(best_dfs)
best_df.to_csv('%s/analysis/stacking-%s.csv' % (path, method), index_label = index_labels)
print '%.3f' % best_df.auc.mean()
