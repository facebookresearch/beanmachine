/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Cumulative sum of the given data.
 *
 * @param {number[]} data - Any array of data.
 * @returns {number[]} The cumulative sum of the given data.
 */
export const cumulativeSum = (data: number[]): number[] => {
  // eslint-disable-next-line arrow-body-style
  const cumulativeSumMap = ((sum: number) => (value: number) => {
    // eslint-disable-next-line no-return-assign
    return (sum += value);
  })(0);

  const cSum = data.map(cumulativeSumMap);
  const normalizationFactor = Math.max(...cSum);
  const cumSum = [];
  for (let i = 0; i < cSum.length; i += 1) {
    cumSum.push(cSum[i] / normalizationFactor);
  }
  return cumSum;
};

/**
 * Numerically sort the given array of numbers.
 *
 * @param {number[]} data - The array of numbers to sort.
 * @returns {number[]} The sorted array of numbers.
 */
export const numericalSort = (data: number[]): number[] => {
  const dataCopy = data.slice(0);
  return dataCopy.sort((a, b) => {
    return a < b ? -1 : a > b ? 1 : 0;
  });
};
