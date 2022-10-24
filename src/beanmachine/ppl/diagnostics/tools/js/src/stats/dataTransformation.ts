/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {argSort, valueCounts} from './array';

/**
 * Scale the given array of numbers by the given scaleFactor. Note that this method
 * divides values in the given array by the scaleFactor.
 *
 * @param {number[]} data - An array of numbers that needs to be scaled.
 * @param {number} scaleFactor - The value to divide all array values by.
 * @returns {number[]} The scaled array.
 */
export const scaleBy = (data: number[], scaleFactor: number): number[] => {
  const scaledData = data.map((datum) => {
    return datum / scaleFactor;
  });
  return scaledData;
};

/**
 * Scale the given array to a maximum value of 1. This method uses the scaleBy method,
 * and finds the maximum value of the given array to use as the scaleFactor of the
 * scaleBy method.
 *
 * @param {number[]} data - Array of data to scale to one.
 * @returns {number[]} The data scaled to one.
 */
export const scaleToOne = (data: number[]): number[] => {
  const scaleFactor = Math.max(...data);
  return scaleBy(data, scaleFactor);
};

/**
 * Assign ranks to the given data. Follows SciPy's and ArviZ's implementations.
 *
 * @param {number[]} data - The numeric data to rank.
 * @returns {number[]} An array of rankings.
 */
export const rankData = (data: number[]): number[] => {
  const n = data.length;
  const rank = Array(n);
  const sortedIndex = argSort(data);
  for (let i = 0; i < rank.length; i += 1) {
    rank[sortedIndex[i]] = i + 1;
  }
  const counts = valueCounts(data);
  const countsArray = Object.entries(counts);
  const keys = [];
  const keyCounts = [];
  for (let i = 0; i < countsArray.length; i += 1) {
    const [key, count] = countsArray[i];
    if (count > 1) {
      keys.push(parseFloat(key));
      keyCounts.push(count);
    }
  }
  for (let i = 0; i < keys.length; i += 1) {
    const repeatIndices = [];
    for (let j = 0; j < data.length; j += 1) {
      if (data[j] === keys[i]) {
        repeatIndices.push(j);
      }
    }
    const rankValues = [];
    for (let k = 0; k < repeatIndices.length; k += 1) {
      rankValues.push(rank[repeatIndices[k]]);
    }
    const sum = rankValues.reduce((previousValue, currentValue) => {
      return previousValue + currentValue;
    }, 0.0);
    const rankMean = sum / rankValues.length;
    for (let k = 0; k < repeatIndices.length; k += 1) {
      rank[repeatIndices[k]] = rankMean;
    }
  }
  return rank;
};
