/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {linearRange, quantile, shape, splitArray, stackArrays} from './array';
import {zScale as zScaleXForm} from './dataTransformation';
import {mean as computeMean} from './pointStatistic';
import {autocovariance, variance1d} from './variance';

/**
 * Compute the effective sample size (ESS) of the given 2D data.
 *
 * @param {number[][]} data - The sample data.
 * @returns {number} Computed ESS of the given data.
 */
const computeESS = (data: number[][]): number => {
  const [numChains, numDraws] = shape(data);
  const chainMeans = [];
  const chainAutocovariances = [];
  const chainVarianceFirstDraws = [];
  const chainVarianceSecondDraws = [];
  for (let i = 0; i < numChains; i += 1) {
    const chainData = data[i];
    const chainMean = computeMean(chainData);
    chainMeans.push(chainMean);
    const chainAutocovariance = autocovariance(chainData);
    chainAutocovariances.push(chainAutocovariance);
    const [chainVarianceFirstDraw, chainVarianceSecondDraw] = chainAutocovariance.slice(
      0,
      2,
    );
    chainVarianceFirstDraws.push(chainVarianceFirstDraw);
    chainVarianceSecondDraws.push(chainVarianceSecondDraw);
  }
  const c = numDraws / (numDraws - 1);
  const meanVarianceFirstDraw = c * computeMean(chainVarianceFirstDraws);
  let variancePlus = (1.0 / c) * meanVarianceFirstDraw;
  if (numChains > 1) {
    variancePlus += variance1d(chainMeans, 1);
  }
  const rhoHat = Array(numDraws).fill(0.0);
  let rhoHatEven = 1.0;
  rhoHat[0] = rhoHatEven;
  const meanVarianceSecondDraw = computeMean(chainVarianceSecondDraws);
  let rhoHatOdd = 1.0 - (meanVarianceFirstDraw - meanVarianceSecondDraw) / variancePlus;
  rhoHat[1] = rhoHatOdd;
  // Geyer's initial positive sequence.
  let t = 1;
  while (t < numDraws - 3 && rhoHatEven + rhoHatOdd > 0.0) {
    const acovEven = [];
    const acovOdd = [];
    for (let i = 0; i < numChains; i += 1) {
      acovEven.push(chainAutocovariances[i][t + 1]);
      acovOdd.push(chainAutocovariances[i][t + 2]);
    }
    const acovEvenMean = computeMean(acovEven);
    const acovOddMean = computeMean(acovOdd);
    rhoHatEven = 1.0 - (meanVarianceFirstDraw - acovEvenMean) / variancePlus;
    rhoHatOdd = 1.0 - (meanVarianceFirstDraw - acovOddMean) / variancePlus;
    if (rhoHatEven + rhoHatOdd >= 0) {
      rhoHat[t + 1] = rhoHatEven;
      rhoHat[t + 2] = rhoHatOdd;
    }
    t += 2;
  }
  // Improve the initial estimation.
  const maxT = t - 2;
  if (rhoHatEven > 0) {
    rhoHat[maxT + 1] = rhoHatEven;
  }
  // Geyer's initial monotone sequence.
  t = 1;
  while (t <= maxT - 2) {
    if (rhoHat[t + 1] + rhoHat[t + 2] > rhoHat[t - 1] + rhoHat[t]) {
      rhoHat[t + 1] = rhoHat[t - 1] + 0.5 * rhoHat[t];
      rhoHat[t + 2] = rhoHat[t + 1];
    }
    t += 2;
  }
  let ess = numChains * numDraws;
  let tauHat =
    -1 +
    2 *
      rhoHat
        .slice(0, maxT + 1)
        // eslint-disable-next-line no-sequences, arrow-body-style
        .reduce((previousValue, currentValue) => previousValue + currentValue, 0) +
    rhoHat
      .slice(maxT + 1, maxT + 2)
      // eslint-disable-next-line no-sequences, arrow-body-style
      .reduce((previousValue, currentValue) => previousValue + currentValue, 0);
  tauHat = Math.max(...[tauHat, 1 / Math.log10(ess)]);
  ess /= tauHat;
  const anyNaNs = rhoHat
    .map((value) => {
      return Number.isNaN(value);
    })
    .reduce(
      // eslint-disable-next-line no-sequences, arrow-body-style
      (previousValue, currentValue) => Number(previousValue) + Number(currentValue),
      0,
    );
  if (anyNaNs) {
    ess = NaN;
  }
  return ess;
};

/**
 * Compute the effective sample size (ESS) quantile. Follows the implementation in ArviZ
 * closely. See
 * https://github.com/arviz-devs/arviz/blob/2d886383019eb95ea9174ea39070c15d03ffdb7e/arviz/stats/diagnostics.py#L739
 * for a more detailed description.
 *
 * @param {number[][]} data - Input data.
 * @param {number} probability - The quantile to compute.
 * @returns {number} The ESS at the given quantile (probability).
 */
export const essQuantile = (data: number[][], probability: number): number => {
  // From SciPy.
  const quantile1d = quantile(data, probability);
  // arviz.stats.diagnostics._ess_quantile
  const iquantile = [];
  const [numChains, numDraws] = shape(data);
  for (let i = 0; i < numChains; i += 1) {
    const chainData = data[i];
    const ichainData = [];
    for (let j = 0; j < numDraws; j += 1) {
      const datum = chainData[j];
      if (datum <= quantile1d) {
        ichainData.push(1);
      } else {
        ichainData.push(0);
      }
    }
    iquantile.push(ichainData);
  }
  // Split each chain in half.
  const splitChains: number[][][] = [];
  for (let i = 0; i < numChains; i += 1) {
    const chainData: number[] = iquantile[i];
    const splitChain: number[][] = splitArray(chainData);
    splitChains.push(splitChain);
  }
  // Stack the chains the way ArviZ does.
  const stackedChains: number[][] = stackArrays(splitChains);
  return computeESS(stackedChains);
};

/**
 * Compute the bulk effective sample size.
 *
 * @param {number[][]} data - Samples from the model.
 * @returns {number} Computed bulk ESS.
 */
export const bulk = (data: number[][]): number => {
  const [numChains] = shape(data);
  // Split each chain in half.
  const splitChains: number[][][] = [];
  for (let i = 0; i < numChains; i += 1) {
    const chainData = data[i];
    const splitChain = splitArray(chainData);
    splitChains.push(splitChain);
  }
  // Stack the chains the way ArviZ does.
  const stackedChains = stackArrays(splitChains);
  // Scale each split chain.
  const zScaled = zScaleXForm(stackedChains);
  // Calculate the Effective Sample Size (ESS).
  return computeESS(zScaled);
};

/**
 * Compute the tail effective sample size.
 *
 * @param {number[][]} data - Samples from the model.
 * @returns {number} Computed tail ESS.
 */
export const tail = (
  data: number[][],
  lowProbability: number = 0.05,
  highProbability: number = 0.95,
): number => {
  const essLowProbabilityQuantile = essQuantile(data, lowProbability);
  const essHighProbabilityQuantile = essQuantile(data, highProbability);
  return Math.min(...[essLowProbabilityQuantile, essHighProbabilityQuantile]);
};

/**
 * Compute the evolution of the effective sample size (ESS) for the given method.
 *
 * @param {number[][]} data - Random variable data.
 * @param {string} method - Can be one of `bulk` or `tail`.
 * @param {number} [numPoints] - The number of points to calculate the evolution for.
 * @returns {{x: number[]; y: number[]}} The ESS evolution over successive draws.
 */
const essEvolution = (
  data: number[][],
  method: string,
  numPoints: number = 20,
): {x: number[]; y: number[]} => {
  const [numChains, numDraws] = shape(data);
  const numSamples = numChains * numDraws;
  const firstDrawIndex = 0;
  const x = linearRange(numSamples / numPoints, numSamples, 1, false, numPoints);
  const drawDivisions = linearRange(
    Math.floor(numDraws / numPoints),
    numDraws,
    1,
    false,
    numPoints,
  );
  const essEvolutionData = [];
  for (let i = 0; i < drawDivisions.length; i += 1) {
    const chainData: number[][] = [];
    for (let j = 0; j < numChains; j += 1) {
      chainData.push(data[j].slice(firstDrawIndex, firstDrawIndex + drawDivisions[i]));
    }
    if (method === 'bulk') {
      essEvolutionData.push(bulk(chainData));
    } else if (method === 'tail') {
      essEvolutionData.push(tail(chainData));
    }
  }
  return {x: x, y: essEvolutionData};
};

/**
 * Syntactic sugar around computing the evolution of the effective sample size (ESS).
 * This method calculates the bulk ESS for the given data.
 *
 * @param {number[][]} data - Random variable data.
 * @returns {{x: number[]; y: number[]}} Bulk effective sample size evolution.
 */
export const essBulkEvolution = (data: number[][]): {x: number[]; y: number[]} => {
  return essEvolution(data, 'bulk');
};

/**
 * Syntactic sugar around computing the evolution of the effective sample size (ESS).
 * This method calculates the tail ESS for the given data.
 *
 * @param {number[][]} data - Random variable data.
 * @returns {{x: number[]; y: number[]}} Tail effective sample size evolution.
 */
export const essTailEvolution = (data: number[][]): {x: number[]; y: number[]} => {
  return essEvolution(data, 'tail');
};
