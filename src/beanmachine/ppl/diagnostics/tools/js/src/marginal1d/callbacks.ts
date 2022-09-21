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

const FIGURENAMES = {marginal: 'marginal', cumulative: 'cumulative'};
const figureNames = Object.keys(FIGURENAMES);

export const updateAxisLabel = (axis: Axis, label: string | null): void => {
  if ('axis_label' in axis) {
    axis.axis_label = label;
  }
};

export const computeStats = (
  rawData: number[],
  computedX: number[],
  computedY: number[],
  hdiProbability: number,
  text_align: string[] = ['right', 'center', 'left'],
  x_offset: number[] = [-5, 0, 5],
  y_offset: number[] = [0, 10, 0],
): interfaces.Marginal1dStats => {
  const mean = computeMean(computedX);
  const hdiBounds = hdiInterval(rawData, hdiProbability);
  const x = [hdiBounds.lowerBound, mean, hdiBounds.upperBound];
  const y = interpolatePoints({x: computedX, y: computedY, points: x});
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

export const computeData = (
  data: number[],
  bwFactor: number,
  hdiProbability: number,
): interfaces.Marginal1dData => {
  const output = {} as interfaces.Marginal1dData;
  for (let i = 0; i < figureNames.length; i += 1) {
    const figureName = figureNames[i];
    output[figureName] = {} as interfaces.Marginal1dDatum;
    const distribution = oneD(data, bwFactor);
    switch (figureName) {
      case 'cumulative':
        distribution.y = scaleToOne(cumulativeSum(distribution.y));
        break;
      default:
        break;
    }
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
    updateAxisLabel(figures[figureName].below[0], rvName);
  }
  return computedData.marginal.distribution.bandwidth;
};
