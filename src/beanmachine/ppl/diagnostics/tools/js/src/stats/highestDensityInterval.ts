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

/**
 * Find the lower and upper bound of the Highest Density Interval (HDI) for the given
 * data.
 *
 * @param {number[]} data - Raw random variable data from the model.
 * @param {number} hdiProbability - The highest density interval probability to use when
 *     calculating the HDI.
 * @returns {HdiInterval} An object defining the lower and upper bound for the HDI of
 *     the given data, as well as the indices of the lower and upper bound for the
 *     sorted raw data.
 */
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

/**
 * Constructs x and y arrays from the HDI bounds using the raw random variable data. It
 * also returns the actual HDI bounds of the random variable data.
 *
 * @param {number[]} rvData - Raw random variable data from the model.
 * @param {number[]} marginalX - The support of the Kernel Density Estimate of the
 *     random variable.
 * @param {number[]} marginalY - The Kernel Density Estimate of the random variable.
 * @param {number} hdiProbability - The highest density interval probability to use when
 *     calculating the HDI.
 * @returns {HdiData} The x array for the HDI data is the "lower" key, while the y array
 *     for the HDI data is the "upper" key and the "base" key is the support along the
 *     axis the HDI is calculated. This nomenclature mirrors the Bokeh Band object,
 *     which is the annotation type we will use when rendering the HDI intervals.
 */
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
  // NOTE: We mirror the Bokeh Band annotation object key names below. The lower key is
  //       an array of zeros since we will fill the Band all the way to the lower axis.
  return {
    base: base,
    lower: new Array(base.length).fill(0.0),
    upper: upper,
    lowerBound: hdi.lowerBound,
    upperBound: hdi.upperBound,
  };
};
