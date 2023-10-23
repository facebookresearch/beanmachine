/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {Axis} from '@bokehjs/models/axes/axis';
import {arrayMean, linearRange} from '../stats/array';
import {
  data as computeHdiData,
  data90Degrees,
  interval as hdiInterval,
} from '../stats/highestDensityInterval';
import {oneD} from '../stats/marginal';
import {interpolatePoints} from '../stats/utils';
import * as interfaces from './interfaces';
import {updateAxisLabel} from '../utils/plottingUtils';

export const computeXData = (
  data: number[][],
  hdiProbability: number,
  bwFactor: number,
): interfaces.XData => {
  const flatData = data.flat();

  // Distribution
  const distribution = oneD(flatData, bwFactor);

  // HDI
  const hdi = computeHdiData(flatData, distribution.x, distribution.y, hdiProbability);
  const hdiData = {base: hdi.base, lower: hdi.lower, upper: hdi.upper};

  // Stats
  const mean = arrayMean(flatData);
  const hdiBounds = hdiInterval(flatData, hdiProbability);
  const x = [hdiBounds.lowerBound, mean, hdiBounds.upperBound];
  const y = interpolatePoints({x: distribution.x, y: distribution.y, points: x});
  const text = [
    `Lower HDI: ${hdiBounds.lowerBound.toFixed(3)}`,
    `Mean: ${mean.toFixed(3)}`,
    `Upper HDI: ${hdiBounds.upperBound.toFixed(3)}`,
  ];
  const output = {
    distribution: distribution,
    hdi: hdiData,
    stats: {x: x, y: y, text: text},
  };

  return output;
};

export const computeYData = (
  data: number[][],
  hdiProbability: number,
  bwFactor: number,
): interfaces.YData => {
  const flatData = data.flat();

  // Distribution
  const distribution = oneD(flatData, bwFactor);

  // HDI
  const hdi = data90Degrees(flatData, distribution.x, distribution.y, hdiProbability);
  const hdiData = {
    lower: {base: hdi.upper.base, lower: hdi.upper.lower, upper: hdi.upper.upper},
    upper: {base: hdi.lower.base, lower: hdi.lower.lower, upper: hdi.lower.upper},
  };

  // Stats
  const mean = arrayMean(flatData);
  const hdiBounds = hdiInterval(flatData, hdiProbability);
  const x = [hdiBounds.lowerBound, mean, hdiBounds.upperBound];
  const y = interpolatePoints({x: distribution.x, y: distribution.y, points: x});
  const text = [
    `Lower HDI: ${hdiBounds.lowerBound.toFixed(3)}`,
    `Mean: ${mean.toFixed(3)}`,
    `Upper HDI: ${hdiBounds.upperBound.toFixed(3)}`,
  ];
  const output = {
    distribution: distribution,
    hdi: hdiData,
    stats: {x: x, y: y, text: text},
  };

  return output;
};

export const computeXYData = (
  rawX: number[][],
  computedX: interfaces.XData,
  rawY: number[][],
  computedY: interfaces.YData,
): interfaces.XYData => {
  const flatDataX = rawX.flat();
  const flatDataY = rawY.flat();

  // NOTE: Falling back to displaying data from the samples as the 2D KDE is not
  //       rendering properly.
  const dataDistribution = {x: flatDataX, y: flatDataY};

  // Stats: Create the stats for the 2D marginal. This is just a central point on the
  // figure showing the mean values of both 1D marginals.
  const stats = {
    x: [computedX.stats.x[1]],
    y: [computedY.stats.x[1]],
    text: [
      `Mean: ${computedX.stats.x[1].toFixed(3)}/${computedY.stats.x[1].toFixed(3)}`,
    ],
  };

  // HDI: Create the HDI guide lines in the 2D marginal distribution. These help the
  // user understand how the 2D probability space is affected by changing the HDI
  // regions of the 1D marginals independently.
  const x = linearRange(
    Math.min(...computedX.hdi.base),
    Math.max(...computedX.hdi.base),
    1,
    true,
    100,
  );
  const y = linearRange(
    Math.min(...computedY.hdi.lower.lower),
    Math.max(...computedY.hdi.upper.upper),
    1,
    true,
    100,
  );
  const hdi = {
    x: {
      lower: {x: Array(y.length).fill(Math.min(...computedX.hdi.base)), y: y},
      upper: {x: Array(y.length).fill(Math.max(...computedX.hdi.base)), y: y},
    },
    y: {
      lower: {x: x, y: Array(x.length).fill(Math.min(...computedY.hdi.lower.lower))},
      upper: {x: x, y: Array(x.length).fill(Math.max(...computedY.hdi.upper.upper))},
    },
  };
  const output = {
    distribution: dataDistribution,
    hdi: hdi,
    stats: stats,
  };
  return output;
};

export const computeData = (
  dataX: number[][],
  hdiProbabilityX: number,
  dataY: number[][],
  hdiProbabilityY: number,
  bwFactor: number,
): interfaces.Data => {
  const xData = computeXData(dataX, hdiProbabilityX, bwFactor);
  const yData = computeYData(dataY, hdiProbabilityY, bwFactor);
  const xyData = computeXYData(dataX, xData, dataY, yData);
  return {x: xData, y: yData, xy: xyData};
};

export const update = (
  dataX: number[][],
  hdiProbabilityX: number,
  dataY: number[][],
  hdiProbabilityY: number,
  bwFactor: number,
  xAxisLabel: string,
  yAxisLabel: string,
  figures: interfaces.Figures,
  sources: interfaces.Sources,
  tooltips: interfaces.Tooltips,
  widgets: interfaces.Widgets,
  glyphs: interfaces.Glyphs,
): number[] => {
  const computedData = computeData(
    dataX,
    hdiProbabilityX,
    dataY,
    hdiProbabilityY,
    bwFactor,
  );
  // Update the x figure.
  const xDistribution = {
    x: computedData.x.distribution.x,
    y: computedData.x.distribution.y,
  };
  const bandwidthX = computedData.x.distribution.bandwidth;
  sources.x.distribution.data = xDistribution;
  sources.x.hdi.data = computedData.x.hdi;
  sources.x.stats.data = computedData.x.stats;
  tooltips.x.distribution.tooltips = [[xAxisLabel, '@x']];
  figures.xy.x_range = figures.x.x_range;

  // Update the y figure.
  const yDistribution = {
    x: computedData.y.distribution.y,
    y: computedData.y.distribution.x,
  };
  const bandwidthY = computedData.y.distribution.bandwidth;
  sources.y.distribution.data = yDistribution;
  sources.y.hdi.lower.data = computedData.y.hdi.lower;
  sources.y.hdi.upper.data = computedData.y.hdi.upper;
  const yStats = {
    x: computedData.y.stats.y,
    y: computedData.y.stats.x,
    text: computedData.y.stats.text,
  };
  sources.y.stats.data = yStats;
  tooltips.y.distribution.tooltips = [[yAxisLabel, '@y']];
  figures.xy.y_range = figures.y.y_range;

  // Update the xy figure.
  sources.xy.distribution.data = computedData.xy.distribution;
  tooltips.xy.distribution.tooltips = [
    [xAxisLabel, '@x'],
    [yAxisLabel, '@y'],
  ];
  sources.xy.hdi.x.lower.data = computedData.xy.hdi.x.lower;
  sources.xy.hdi.x.upper.data = computedData.xy.hdi.x.upper;
  tooltips.xy.hdi.x.lower.tooltips = [[xAxisLabel, '@x']];
  tooltips.xy.hdi.x.upper.tooltips = [[xAxisLabel, '@x']];
  sources.xy.hdi.y.lower.data = computedData.xy.hdi.y.lower;
  sources.xy.hdi.y.upper.data = computedData.xy.hdi.y.upper;
  tooltips.xy.hdi.y.lower.tooltips = [[yAxisLabel, '@y']];
  tooltips.xy.hdi.y.upper.tooltips = [[yAxisLabel, '@y']];
  sources.xy.stats.data = computedData.xy.stats;
  tooltips.xy.stats.tooltips = [
    [xAxisLabel, '@x'],
    [yAxisLabel, '@y'],
  ];

  (window as any).data = computedData;
  (window as any).figures = figures;
  (window as any).glyphs = glyphs;
  (window as any).sources = sources;

  updateAxisLabel(figures.xy.below[0] as Axis, xAxisLabel);
  updateAxisLabel(figures.xy.left[0] as Axis, yAxisLabel);

  // Update widgets.
  widgets.bw_div_x.text = `Bandwidth ${xAxisLabel}: ${bwFactor * bandwidthX}`;
  widgets.bw_div_y.text = `Bandwidth ${yAxisLabel}: ${bwFactor * bandwidthY}`;
  return [bandwidthX, bandwidthY];
};
