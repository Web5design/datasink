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

from common import load_properties
from diversity import average_diversity_score
from pandas import DataFrame, concat, read_csv
from sklearn.metrics import mean_squared_error, roc_auc_score

path = abspath(argv[1])
assert exists(path)
if not exists('%s/analysis' % path):
    mkdir('%s/analysis' % path)
p = load_properties(path)
fold_count = int(p['foldCount'])

dfs = []
for fold in range(fold_count):
    df          = read_csv('%s/validation-%s.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
    labels      = df.index.get_level_values(1).values
    predictions = df.mean(axis = 1)
    auc         = roc_auc_score(labels, predictions)
    brier       = mean_squared_error(labels, predictions)
    diversity   = average_diversity_score(df.values)
    dfs.append(DataFrame({'auc': auc, 'brier': brier, 'diversity': diversity}, index = [fold]))
perf_df = concat(dfs)
perf_df.to_csv('%s/analysis/mean.csv' % path, index_label = 'fold')
print '%.3f' % perf_df.auc.mean()
