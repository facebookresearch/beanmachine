// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

/**
 * Syntactic sugar for summing an array of numbers.
 *
 * @param {number[]} data - The array of data.
 * @returns {number} The sum of the array of data.
 */
export const sum = (data: number[]): number => {
  return data.reduce((previousValue, currentValue) => {
    return previousValue + currentValue;
  });
};

/**
 * Calculate the mean of the given array of data.
 *
 * @param {number[]} data - The array of data.
 * @returns {number} The mean of the given data.
 */
export const mean = (data: number[]): number => {
  const dataSum = sum(data);
  return dataSum / data.length;
};
