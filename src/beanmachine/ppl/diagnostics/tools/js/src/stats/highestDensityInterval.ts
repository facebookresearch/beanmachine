// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {numericalSort} from './array';

export interface HdiInterval {
  lowerBound: number;
  upperBound: number;
  lowerBoundIndex: number;
  upperBoundIndex: number;
}

export interface HdiData {
  base: number[];
  lower: number[];
  upper: number[];
  lowerBound: number;
  upperBound: number;
}

export const interval = (data: number[], hdiProbability: number): HdiInterval => {
  const N = data.length;
  const sortedData = numericalSort(data);
  const stopIndex = Math.floor(hdiProbability * N);
  const startIndex = N - stopIndex;
  const leftData = sortedData.slice(stopIndex);
  const rightData = sortedData.slice(0, startIndex);
  const hdi = [];
  for (let i = 0; i < leftData.length; i += 1) {
    hdi.push(leftData[i] - rightData[i]);
  }
  const lowerIndex = hdi.indexOf(Math.min(...hdi));
  const upperIndex = lowerIndex + stopIndex;
  return {
    lowerBound: sortedData[lowerIndex],
    upperBound: sortedData[upperIndex],
    lowerBoundIndex: lowerIndex,
    upperBoundIndex: upperIndex,
  };
};

export const data = (
  rvData: number[],
  marginalX: number[],
  marginalY: number[],
  hdiProbability: number,
): HdiData => {
  const hdi = interval(rvData, hdiProbability);
  const base = [];
  const upper = [];
  for (let i = 0; i < marginalX.length; i += 1) {
    if (marginalX[i] <= hdi.upperBound && marginalX[i] >= hdi.lowerBound) {
      base.push(marginalX[i]);
      upper.push(marginalY[i]);
    }
  }
  return {
    base: base,
    lower: new Array(base.length).fill(0.0),
    upper: upper,
    lowerBound: hdi.lowerBound,
    upperBound: hdi.upperBound,
  };
};
