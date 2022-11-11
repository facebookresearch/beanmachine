/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

/**
 * Computes the inverse error function at the given point. See
 * https://en.wikipedia.org/wiki/Error_function#Inverse_functions.
 *
 * @param {number} x - The point to calculate the inverse error function of.
 * @returns {number} The value of the inverse error function.
 */
export const inverseERF = (x: number): number => {
  // maximum relative error = .00013
  const a = 0.147;
  const b = 2 / (a * Math.PI) + 0.5 * Math.log(1 - x ** 2);
  const c = Math.log(1 - x ** 2) / a;
  const d = Math.sqrt(b ** 2 - c);
  const e = Math.sqrt(d - b);
  return Math.sign(x) * e;
};

/**
 * Compute the percent point function (PPF) at the given point. See
 * https://en.wikipedia.org/wiki/Quantile_function.
 *
 * @param {number} x - The point where to calculate the PPF.
 * @returns {number} The calculated PPF.
 */
export const ppf = (x: number): number => {
  return Math.sqrt(2) * inverseERF(2 * x - 1);
};
