/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {density1d} from 'fast-kde/src/density1d';
import {density2d} from 'fast-kde/src/density2d';
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

/**
 * Computes the 2D Kernel Density Estimate.
 *
 * @param {number[]} x - The raw random variable data of the model in the x direction.
 * @param {number[]} y - The raw random variable data of the model in the y direction.
 * @param {number} bwFactor - Multiplicative factor to be applied to the bandwidth when
 *     calculating the Kernel Density Estimate (KDE).
 * @param {number[]} [bins] - The number of bins to use for calculating the 2D KDE.
 * @returns {{x: number[]; y: number[]; z: number[]; bw: {x: number; y: number}}}
 *     The computed 2D KDE with bandwidths for both sets of data.
 */
export const twoD = (
  x: number[],
  y: number[],
  bwFactor: number,
  bins: number[] = [128, 128],
): {x: number[]; y: number[]; z: number[]; bw: {x: number; y: number}} => {
  // Prepare the random variables for calculating the 2D KDE using fast-kde.
  const data = [];
  for (let i: number = 0; i < x.length; i += 1) {
    data.push({u: x[i], v: y[i]});
  }

  // Calculate the 2D KDE.
  const kde2d = density2d(data, {x: 'u', y: 'v', bins: bins, adjust: bwFactor, pad: 3});
  const [bwX, bwY] = kde2d.bandwidth();

  // Extract the 2D data points from the 2D KDE calculation.
  const points: {x: number; y: number; z: number}[] = [...kde2d];
  const X: number[] = [];
  const Y: number[] = [];
  const Z: number[] = [];
  for (let i: number = 0; i < points.length; i += 1) {
    X[i] = points[i].x;
    Y[i] = points[i].y;
    Z[i] = points[i].z;
  }

  return {x: X, y: Y, z: Z, bw: {x: bwX, y: bwY}};
};
