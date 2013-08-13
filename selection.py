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

from common import get_best_performer, load_properties, read_fold
from diversity import average_diversity_score
from numpy import array
from numpy.random import choice, seed
from pandas import DataFrame, concat
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, roc_auc_score

def eval_metrics(df, ensemble, labels, indices, final = False):
    predictions     = df[ensemble].mean(axis = 1)
    auc             = roc_auc_score(labels, predictions)
    brier           = mean_squared_error(labels, predictions)
    diversity       = average_diversity_score(df[ensemble].values)
    ensemble_size   = len(ensemble)
    ensemble        = ' '.join(ensemble) if final else ensemble[-1]
    return DataFrame({'auc': auc, 'brier': brier, 'diversity': diversity, 'ensemble': ensemble, 'ensemble_size': ensemble_size}, index = indices)


def select_candidate_greedy(train_df, train_labels, best_classifiers, ensemble, i):
    return best_classifiers.index.values[i]


def select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i):
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)
        candidate_scores = [roc_auc_score(train_labels, train_df[ensemble + [candidate]].mean(axis = 1)) for candidate in candidates]
        best_candidate = candidates[array(candidate_scores).argmax()]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def select_candidate_drep(train_df, train_labels, best_classifiers, ensemble, i):
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)
        candidate_diversity_scores = [abs(average_diversity_score(train_df[ensemble + [candidate]].values)) for candidate in candidates]
        candidate_diversity_ranks = array(candidate_diversity_scores).argsort()
        diversity_candidates = candidates[candidate_diversity_ranks[:max_diversity_candidates]]
        candidate_accuracy_scores = [roc_auc_score(train_labels, train_df[ensemble + [candidate]].mean(axis = 1)) for candidate in diversity_candidates]
        best_candidate = candidates[array(candidate_accuracy_scores).argmax()]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def select_candidate_sdi(train_df, train_labels, best_classifiers, ensemble, i):
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)
        candidate_diversity_scores = [1 - abs(average_diversity_score(train_df[ensemble + [candidate]].values)) for candidate in candidates] # 1 - kappa so larger = more diverse
        candidate_scores = [accuracy_weight * best_classifiers.ix[candidate] + (1 - accuracy_weight) * candidate_diversity_scores[candidate_i] for candidate_i, candidate in enumerate(candidates)]
        best_candidate = candidates[array(candidate_scores).argmax()]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def selection(fold):
    seed(seedval)
    indices = [[fold], [seedval]]
    train_df, train_labels, test_df, test_labels = read_fold(path, fold)
    best_classifiers = train_df.apply(lambda x: roc_auc_score(train_labels, x)).order(ascending = False)
    train_metrics = []
    test_metrics = []
    ensemble = []
    for i in range(min(max_ensemble_size, len(best_classifiers))):
        best_candidate = select_candidate(train_df, train_labels, best_classifiers, ensemble, i)
        ensemble.append(best_candidate)
        train_metrics.append(eval_metrics(train_df, ensemble, train_labels, indices))
        test_metrics.append(eval_metrics(test_df, ensemble, test_labels, indices))
    train_metrics_df = concat(train_metrics)
    best_ensemble_size = get_best_performer(train_metrics_df).ensemble_size
    best_ensemble = train_metrics_df.ensemble[:best_ensemble_size + 1]
    return eval_metrics(test_df, best_ensemble, test_labels, indices, final = True), concat(test_metrics)


path = abspath(argv[1])
assert exists(path)
if not exists('%s/analysis' % path):
    mkdir('%s/analysis' % path)
method = argv[2]
assert method in set(['greedy', 'enhanced', 'drep', 'sdi'])
select_candidate = eval('select_candidate_' + method)
p = load_properties(path)
fold_count = int(p['foldCount'])
initial_ensemble_size = 2
max_ensemble_size = 50
max_candidates = 50
max_diversity_candidates = 5
accuracy_weight = 0.5

index_labels = ['fold', 'seed']
best_dfs = []
iteration_dfs = []
seeds = range(10) if method in set(['enhanced', 'drep', 'sdi']) else [0]
for seedval in seeds:
    results = Parallel(n_jobs = -1, verbose = 0)(delayed(selection)(fold) for fold in range(fold_count))
    for best_df, iteration_df in results:
        best_dfs.append(best_df)
        iteration_dfs.append(iteration_df)
iteration_df = concat(iteration_dfs)
iteration_df.to_csv('%s/analysis/selection-%s-iterations.csv' % (path, method), index_label = index_labels)
best_df = concat(best_dfs)
best_df.to_csv('%s/analysis/selection-%s.csv' % (path, method), index_label = index_labels)
print '%.3f %i' % (best_df.auc.mean(), best_df.ensemble_size.mean())
