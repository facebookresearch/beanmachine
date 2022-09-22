// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {density1d} from 'fast-kde/src/density1d';
import {scaleToOne} from './dataTransformation';

/**
 * Calculate the one-dimensional Kernel Density Estimate. This method uses the "scott"
 * estimate for the bandwidth used when calculating the marginal. This bandwidth
 * estimate is the same as is used in the corresponding Python KDE estimate from ArviZ.
 *
 * Scott, D (1979). On optimal and data-based histograms. Biometrika. 66 (3): 605–610.
 * doi:10.1093/biomet/66.3.605.
 *
 * @param {number[]} data - The raw random variable data of the model.
 * @param {number} bwFactor - Multiplicative factor to be applied to the bandwidth when
 *     calculating the Kernel Density Estimate (KDE).
 * @returns {{x: number[]; y: number[]; bandwidth: number}} The KDE of the given data,
 *     along with the numerical value of the bandwidth used to calculate the KDE.
 */
// eslint-disable-next-line import/prefer-default-export
export const oneD = (
  data: number[],
  bwFactor: number,
): {x: number[]; y: number[]; bandwidth: number} => {
  // Calculate the 1D KDE.
  const kde1d = density1d(data, {
    adjust: bwFactor,
    pad: 3,
    bins: 512,
  });
  // Extract the data points from the 1D KDE calculation.
  const points: {x: number; y: number}[] = Array.from(kde1d);
  const x = [];
  const y = [];
  for (let i = 0; i < points.length; i += 1) {
    x.push(points[i].x);
    y.push(points[i].y);
  }
  return {
    x: x,
    y: scaleToOne(y),
    bandwidth: kde1d.bandwidth(),
  };
};
