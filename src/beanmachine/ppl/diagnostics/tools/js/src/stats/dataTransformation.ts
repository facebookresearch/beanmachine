/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {argSort, createEmpty2dArray, shape, valueCounts} from './array';
import {mean, ppf} from './pointStatistic';

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

/**
 * Translate the given data such that it is centered about its mean.
 *
 * @param {number[]} data - Data needing to be translated.
 * @returns {number[]} The translated data.
 */
export const translateToMean = (data: number[]): number[] => {
  const dataMean = mean(data);
  const translation = data.map((datum) => {
    return datum - dataMean;
  });
  return translation;
};

/**
 * Back transformation of ranks. The fractional offset defaults to 3/8 as recommended by
 * Blom (1958). This method follows ArviZ's implementation closely.
 *
 * Blom G (1958). Statistical Estimates and Transformed Beta-Variables. Wiley; New York.
 *
 * @param {number[]} data - Rank data.
 * @param {number} [fractionalOffset] - Fractional offset, defaults to 3/8.
 * @returns {number[]} The back transformation of the given rank data.
 */
export const backTransformRanks = (
  data: number[],
  fractionalOffset: number = 3 / 8,
): number[] => {
  const n = data.length;
  const backXForm = [];
  for (let i = 0; i < data.length; i += 1) {
    const numerator = data[i] - fractionalOffset;
    const denominator = n - 2 * fractionalOffset + 1;
    backXForm.push(numerator / denominator);
  }
  return backXForm;
};

/**
 * Compute the z-scale of the data, by first ranking it, then back transforming it.
 *
 * @param {number[][]} data - Data to compute.
 * @returns {number[][]} Scaled data.
 */
export const zScale = (data: number[][]): number[][] => {
  const [numRows, numColumns] = shape(data);
  const flatData = data.flat();
  // Compute the ranks of the data.
  let rank = rankData(flatData);
  // Back transform the ranks data.
  rank = backTransformRanks(rank);
  // Compute scaled back transformed ranks data.
  const z = [];
  for (let i = 0; i < rank.length; i += 1) {
    z.push(ppf(rank[i]));
  }
  // Reshape the scaled data back to the original data shape.
  const reshapedZ = createEmpty2dArray(numRows, numColumns);
  let k = 0;
  for (let i = 0; i < numRows; i += 1) {
    for (let j = 0; j < numColumns; j += 1) {
      reshapedZ[i][j] = z[k];
      k += 1;
    }
  }
  return reshapedZ;
};
