// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {Axis} from '@bokehjs/models/axes/axis';
import {cumulativeSum} from '../stats/array';
import {scaleToOne} from '../stats/dataTransformation';
import {
  interval as hdiInterval,
  data as hdiData,
} from '../stats/highestDensityInterval';
import {oneD} from '../stats/marginal';
import {mean as computeMean} from '../stats/pointStatistic';
import {interpolatePoints} from '../stats/utils';
import * as interfaces from './interfaces';

// Define the names of the figures used for this Bokeh application.
const figureNames = ['marginal', 'cumulative'];

/**
 * Update the given Bokeh Axis object with the new label string. You must use this
 * method to update axis strings using TypeScript, otherwise the ts compiler will throw
 * a type check error.
 *
 * @param {Axis} axis - The Bokeh Axis object needing a new label.
 * @param {string | null} label - The new label for the Bokeh Axis object.
 */
export const updateAxisLabel = (axis: Axis, label: string | null): void => {
  // Type check requirement.
  if ('axis_label' in axis) {
    axis.axis_label = label;
  }
};

/**
 * Compute the following statistics for the given random variable data
 *
 * - lower bound for the highest density interval calculated from the marginalX;
 * - mean of the rawData;
 * - upper bound for the highest density interval calculated from the marginalY.
 *
 * @param {number[]} rawData - Raw random variable data from the model.
 * @param {number[]} marginalX - The support of the Kernel Density Estimate of the
 *     random variable.
 * @param {number[]} marginalY - The Kernel Density Estimate of the random variable.
 * @param {number | null} [hdiProb=null] - The highest density interval probability
 *     value. If the default value is not overwritten, then the default HDI probability
 *     is 0.89. See Statistical Rethinking by McElreath for a description as to why this
 *     value is the default.
 * @param {string[]} [text_align=['right', 'center', 'left']] - How to align the text
 *     shown in the figure for the point statistics.
 * @param {number[]} [x_offset=[-5, 0, 5]] - Offset values for the text along the
 *     x-axis.
 * @param {number[]} [y_offset=[0, 10, 0]] - Offset values for the text along the
 *     y-axis
 * @returns {interfaces.Marginal1dStats} Object containing the computed stats.
 */
export const computeStats = (
  rawData: number[],
  marginalX: number[],
  marginalY: number[],
  hdiProb: number | null = null,
  text_align: string[] = ['right', 'center', 'left'],
  x_offset: number[] = [-5, 0, 5],
  y_offset: number[] = [0, 10, 0],
): interfaces.Marginal1dStats => {
  // Set the default value to 0.89 if no default value has been given.
  const hdiProbability = hdiProb ?? 0.89;

  // Compute the point statistics for the KDE, and create labels to display them in the
  // figures.
  const mean = computeMean(marginalX);
  const hdiBounds = hdiInterval(rawData, hdiProbability);
  const x = [hdiBounds.lowerBound, mean, hdiBounds.upperBound];
  const y = interpolatePoints({x: marginalX, y: marginalY, points: x});
  const text = [
    `Lower HDI: ${hdiBounds.lowerBound.toFixed(3)}`,
    `Mean: ${mean.toFixed(3)}`,
    `Upper HDI: ${hdiBounds.upperBound.toFixed(3)}`,
  ];

  return {
    x: x,
    y: y,
    text: text,
    text_align: text_align,
    x_offset: x_offset,
    y_offset: y_offset,
  };
};

/**
 * Compute data for the one-dimensional marginal diagnostic tool.
 *
 * @param {number[]} data - Raw random variable data from the model.
 * @param {number} bwFactor - Multiplicative factor to be applied to the bandwidth when
 *     calculating the Kernel Density Estimate (KDE).
 * @param {number} hdiProbability - The highest density interval probability to use when
 *     calculating the HDI.
 * @returns {interfaces.Marginal1dData} The marginal distribution and cumulative
 *     distribution calculated from the given random variable data. Point statistics are
 *     also calculated.
 */
export const computeData = (
  data: number[],
  bwFactor: number,
  hdiProbability: number,
): interfaces.Marginal1dData => {
  const output = {} as interfaces.Marginal1dData;
  for (let i = 0; i < figureNames.length; i += 1) {
    const figureName = figureNames[i];
    output[figureName] = {} as interfaces.Marginal1dDatum;

    // Compute the one-dimensional KDE and its cumulative distribution.
    const distribution = oneD(data, bwFactor);
    switch (figureName) {
      case 'cumulative':
        distribution.y = scaleToOne(cumulativeSum(distribution.y));
        break;
      default:
        break;
    }

    // Compute the point statistics for the given data.
    const stats = computeStats(data, distribution.x, distribution.y, hdiProbability);

    output[figureName] = {
      distribution: distribution,
      hdi: hdiData(data, distribution.x, distribution.y, hdiProbability),
      stats: {x: stats.x, y: stats.y},
      labels: stats,
    };
  }
  return output;
};

/**
 * Callback used to update the Bokeh application in the notebook.
 *
 * @param {number[]} data - Raw random variable data from the model.
 * @param {string} rvName - The name of the random variable from the model.
 * @param {number} bwFactor - Multiplicative factor to be applied to the bandwidth when
 *     calculating the kernel density estimate.
 * @param {number} hdiProbability - The highest density interval probability to use when
 *     calculating the HDI.
 * @param {interfaces.Marginal1dSources} sources - Bokeh sources used to render glyphs
 *     in the application.
 * @param {interfaces.Marginal1dFigures} figures - Bokeh figures shown in the
 *     application.
 * @returns {number} We display the value of the bandwidth used for computing the Kernel
 *     Density Estimate in a div, and must return that value here in order to update the
 *     value displayed to the user.
 */
export const update = (
  data: number[],
  rvName: string,
  bwFactor: number,
  hdiProbability: number,
  sources: interfaces.Marginal1dSources,
  figures: interfaces.Marginal1dFigures,
): number => {
  const computedData = computeData(data, bwFactor, hdiProbability);
  for (let i = 0; i < figureNames.length; i += 1) {
    // Update all sources with new data calculated above.
    const figureName = figureNames[i];
    sources[figureName].distribution.data = {
      x: computedData[figureName].distribution.x,
      y: computedData[figureName].distribution.y,
    };
    sources[figureName].hdi.data = {
      base: computedData[figureName].hdi.base,
      lower: computedData[figureName].hdi.lower,
      upper: computedData[figureName].hdi.upper,
    };
    sources[figureName].stats.data = {
      x: computedData[figureName].stats.x,
      y: computedData[figureName].stats.y,
      label: computedData[figureName].labels.text,
    };
    sources[figureName].labels.data = computedData[figureName].labels;

    // Update the axes labels.
    updateAxisLabel(figures[figureName].below[0], rvName);
  }
  return computedData.marginal.distribution.bandwidth;
};
