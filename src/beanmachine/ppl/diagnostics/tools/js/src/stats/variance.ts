/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ndarray from 'ndarray';
import ndfft from 'ndarray-fft/fft';
import {translateToMean} from './dataTransformation';

/**
 * Compute the autocovariance of the given data for every lag of the input array.
 *
 * @param {number[]} data - Samples from the model.
 * @returns {number[]} The autocovariance of the given samples.
 */
export const autocovariance = (data: number[]): number[] => {
  // Center the data about the mean.
  const dataCentered = translateToMean(data);
  // Run the forward FFT. This updates the realSignal and imagSignal variables directly.
  const forwardRealSignal = ndarray(dataCentered);
  const forwardImagSignal = ndarray(Array(dataCentered.length).fill(0.0));
  ndfft(1, forwardRealSignal, forwardImagSignal);
  // Square the data, i.e. compute the complex conjugate.
  const dataSquared = [];
  for (let i = 0; i < forwardRealSignal.data.length; i += 1) {
    dataSquared.push(forwardRealSignal.data[i] ** 2 + forwardImagSignal.data[i] ** 2);
  }
  // Calculate the reverse FFT. Again this updates the realSignal and imagSignal
  // variables directly.
  const reverseRealSignal = ndarray(dataSquared);
  const reverseImagSignal = ndarray(Array(dataCentered.length).fill(0.0));
  ndfft(-1, reverseRealSignal, reverseImagSignal);
  // Normalize by dividing by the length of the data.
  const autocov = [];
  const N = data.length;
  for (let i = 0; i < reverseRealSignal.data.length; i += 1) {
    autocov.push(reverseRealSignal.data[i] / N);
  }
  return autocov;
};

/**
 * Compute the 1D variance of the given data.
 *
 * @param {number[]} data - Samples from the model.
 * @param {number} [degreesOfFreedom] - Degrees of freedom.
 * @returns {number} The computed variance.
 */
export const variance1d = (data: number[], degreesOfFreedom: number = 0): number => {
  const n = data.length;
  let total = 0;
  let b = 0;
  for (let i = 0; i < data.length; i += 1) {
    total += data[i];
    b += data[i] ** 2;
  }
  let variance = b / n - (total / n) ** 2;
  variance *= n / (n - degreesOfFreedom);
  return variance;
};
