/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {Plot} from '@bokehjs/models/plots/plot';
import {ColumnDataSource} from '@bokehjs/models/sources/column_data_source';

// NOTE: In the corresponding Python typing files for the diagnostic tool, we define
//       similar types using a TypedDict object. TypeScript allows us to maintain
//       semantic information about the key names and their types in the same object and
//       still have the ability to loop over objects as long as we have the
//       [key: string]: any; indicator in the interface definition. This boils down to a
//       Python type of Dict[Any, Any], which again loses all type information about the
//       object we are defining. We are mirroring what is done in Python here, so we
//       keep the semantic information here at the expense of losing type information
//       similarly to what is done in Python.

export interface Marginal1dStats {
  [key: string]: any;
  x: number[];
  y: number[];
  text: string[];
  text_align: string[];
  x_offset: number[];
  y_offset: number[];
}

// Used for both the marginal and cumulative distribution objects.
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

// Used for both the marginal and cumulative distribution objects.
export interface Marginal1dSource {
  [key: string]: any;
  distribution: ColumnDataSource;
  hdi: ColumnDataSource;
  stats: ColumnDataSource;
  labels: ColumnDataSource;
}

export interface Marginal1dSources {
  [key: string]: any;
  marginal: Marginal1dSource;
  cumulative: Marginal1dSource;
}

export interface Marginal1dFigures {
  [key: string]: any;
  marginal: Plot;
  cumulative: Plot;
}
