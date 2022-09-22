// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

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
