/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {autocovariance} from './variance';
import {scaleBy} from './dataTransformation';

/**
 * Compute the autocorrelation of the given data using FFT for every lag of the array.
 *
 * @param {number[]} data - The data to compute the autocorrelation.
 * @returns {number[]} The autocorrelation of the given data.
 */
// eslint-disable-next-line import/prefer-default-export
export const autocorrelation = (data: number[]): number[] => {
  const acov = autocovariance(data);
  const [scaleFactor] = acov;
  const acor = scaleBy(acov, scaleFactor);
  return acor;
};
