/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {Axis} from '@bokehjs/models/axes/axis';
import {arrayMean, arrayMedian, cumulativeSum} from '../stats/array';
import {scaleToOne} from '../stats/dataTransformation';
import {
  interval as hdiInterval,
  data as hdiData,
} from '../stats/highestDensityInterval';
import {oneD} from '../stats/marginal';
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
 * @param {number} activeStatistic - The statistic to show in the tool. 0 is the mean
 *     and 1 is the median.
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
 * @returns {interfaces.LabelsData} Object containing the computed stats.
 */
export const computeStats = (
  rawData: number[],
  marginalX: number[],
  marginalY: number[],
  activeStatistic: number,
  hdiProb: number | null = null,
  text_align: string[] = ['right', 'center', 'left'],
  x_offset: number[] = [-5, 0, 5],
  y_offset: number[] = [0, 10, 0],
): interfaces.LabelsData => {
  // Set the default value to 0.89 if no default value has been given.
  const hdiProbability = hdiProb ?? 0.89;

  // Compute the point statistics for the KDE, and create labels to display them in the
  // figures.
  const mean = arrayMean(rawData);
  const median = arrayMedian(rawData);
  const hdiBounds = hdiInterval(rawData, hdiProbability);
  let x = [hdiBounds.lowerBound, mean, median, hdiBounds.upperBound];
  let y = interpolatePoints({x: marginalX, y: marginalY, points: x});
  let text = [
    `Lower HDI: ${hdiBounds.lowerBound.toFixed(3)}`,
    `Mean: ${mean.toFixed(3)}`,
    `Median: ${median.toFixed(3)}`,
    `Upper HDI: ${hdiBounds.upperBound.toFixed(3)}`,
  ];

  // We will filter the output based on the active statistic from the tool.
  let mask: number[] = [];
  if (activeStatistic === 0) {
    mask = [0, 1, 3];
  } else if (activeStatistic === 1) {
    mask = [0, 2, 3];
  }
  x = mask.map((i) => {
    return x[i];
  });
  y = mask.map((i) => {
    return y[i];
  });
  text = mask.map((i) => {
    return text[i];
  });

  const output = {
    x: x,
    y: y,
    text: text,
    text_align: text_align,
    x_offset: x_offset,
    y_offset: y_offset,
  };
  return output;
};

/**
 * Compute data for the one-dimensional marginal diagnostic tool.
 *
 * @param {number[]} data - Raw random variable data from the model.
 * @param {number} bwFactor - Multiplicative factor to be applied to the bandwidth when
 *     calculating the Kernel Density Estimate (KDE).
 * @param {number} hdiProbability - The highest density interval probability to use when
 *     calculating the HDI.
 * @param {number} activeStatistic - The statistic to show in the tool. 0 is the mean
 *     and 1 is the median.
 * @returns {interfaces.Data} The marginal distribution and cumulative
 *     distribution calculated from the given random variable data. Point statistics are
 *     also calculated.
 */
export const computeData = (
  data: number[],
  bwFactor: number,
  hdiProbability: number,
  activeStatistic: number,
): interfaces.Data => {
  const output = {} as interfaces.Data;
  for (let i = 0; i < figureNames.length; i += 1) {
    const figureName = figureNames[i];
    output[figureName] = {} as interfaces.GlyphData;

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
    const stats = computeStats(
      data,
      distribution.x,
      distribution.y,
      activeStatistic,
      hdiProbability,
    );

    output[figureName] = {
      distribution: distribution,
      hdi: hdiData(data, distribution.x, distribution.y, hdiProbability),
      stats: {x: stats.x, y: stats.y, text: stats.text},
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
 * @param {interfaces.Sources} sources - Bokeh sources used to render glyphs in the
 *     application.
 * @param {interfaces.Figures} figures - Bokeh figures shown in the application.
 * @param {interfaces.Tooltips} tooltips - Bokeh tooltips shown on the glyphs.
 * @param {interfaces.Widgets} widgets - Bokeh widget object for the tool.
 * @returns {number} We display the value of the bandwidth used for computing the Kernel
 *     Density Estimate in a div, and must return that value here in order to update the
 *     value displayed to the user.
 */
export const update = (
  data: number[],
  rvName: string,
  bwFactor: number,
  hdiProbability: number,
  sources: interfaces.Sources,
  figures: interfaces.Figures,
  tooltips: interfaces.Tooltips,
  widgets: interfaces.Widgets,
): number => {
  const activeStatistic = widgets.stats_button.active as number;
  const computedData = computeData(data, bwFactor, hdiProbability, activeStatistic);

  // Marginal figure.
  // eslint-disable-next-line prefer-destructuring
  const bandwidth = computedData.marginal.distribution.bandwidth;
  sources.marginal.distribution.data = {
    x: computedData.marginal.distribution.x,
    y: computedData.marginal.distribution.y,
  };
  sources.marginal.hdi.data = {
    base: computedData.marginal.hdi.base,
    lower: computedData.marginal.hdi.lower,
    upper: computedData.marginal.hdi.upper,
  };
  sources.marginal.stats.data = computedData.marginal.stats;
  sources.marginal.labels.data = computedData.marginal.labels;
  tooltips.marginal.distribution.tooltips = [[rvName, '@x']];
  tooltips.marginal.stats.tooltips = [['', '@text']];
  updateAxisLabel(figures.marginal.below[0] as Axis, rvName);

  // Cumulative figure.
  sources.cumulative.distribution.data = {
    x: computedData.cumulative.distribution.x,
    y: computedData.cumulative.distribution.y,
  };
  sources.cumulative.hdi.data = {
    base: computedData.cumulative.hdi.base,
    lower: computedData.cumulative.hdi.lower,
    upper: computedData.cumulative.hdi.upper,
  };
  sources.cumulative.stats.data = computedData.cumulative.stats;
  sources.cumulative.labels.data = computedData.cumulative.labels;
  tooltips.cumulative.distribution.tooltips = [[rvName, '@x']];
  tooltips.cumulative.stats.tooltips = [['', '@text']];
  updateAxisLabel(figures.cumulative.below[0] as Axis, rvName);

  return bandwidth;
};
