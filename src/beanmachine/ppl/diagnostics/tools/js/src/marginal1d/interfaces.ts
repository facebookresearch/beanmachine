// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {Plot} from '@bokehjs/models/plots/plot';
import {ColumnDataSource} from '@bokehjs/models/sources/column_data_source';

export interface Marginal1dStats {
  [key: string]: any;
  x: number[];
  y: number[];
  text: string[];
  text_align: string[];
  x_offset: number[];
  y_offset: number[];
}

export interface Marginal1dDatum {
  distribution: {[key: string]: any; x: number[]; y: number[]; bandwidth: number};
  hdi: {
    [key: string]: any;
    base: number[];
    lower: number[];
    upper: number[];
    lowerBound: number;
    upperBound: number;
  };
  stats: {[key: string]: any; x: number[]; y: number[]};
  labels: {
    [key: string]: any;
    x: number[];
    y: number[];
    text: string[];
    text_align: string[];
    x_offset: number[];
    y_offset: number[];
  };
}

export interface Marginal1dData {
  [key: string]: any;
  marginal: Marginal1dDatum;
  cumulative: Marginal1dDatum;
}

export interface Marginal1dSources {
  [key: string]: any;
  marginal: {
    [key: string]: any;
    distribution: ColumnDataSource;
    hdi: ColumnDataSource;
    stats: ColumnDataSource;
    labels: ColumnDataSource;
  };
  cumulative: {
    [key: string]: any;
    distribution: ColumnDataSource;
    hdi: ColumnDataSource;
    stats: ColumnDataSource;
    labels: ColumnDataSource;
  };
}

export interface Marginal1dFigures {
  [key: string]: any;
  marginal: Plot;
  cumulative: Plot;
}
