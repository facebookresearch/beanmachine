/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import {linearRange, shape} from '../stats/array';
import {essBulkEvolution, essTailEvolution} from '../stats/effectiveSampleSize';
import * as interfaces from './interfaces';

export const computeData = (data: number[][]): interfaces.Data => {
  const [numChains, numDraws] = shape(data);
  const numDrawsAllChains = numChains * numDraws;
  const essBulkData = essBulkEvolution(data);
  const numPoints = essBulkData.x.length;
  const ruleOfThumb = 100 * numChains;
  const ruleOfThumbX = linearRange(0, numDrawsAllChains, 1, false, numPoints);
  const ruleOfThumbY = Array(numPoints).fill(ruleOfThumb) as Array<number>;
  const ruleOfThumbLabel = Array(numPoints).fill(
    ruleOfThumb.toString(),
  ) as Array<string>;
  const output = {} as interfaces.Data;
  output.ess = {
    bulk: essBulkData,
    tail: essTailEvolution(data),
    ruleOfThumb: {x: ruleOfThumbX, y: ruleOfThumbY, text: ruleOfThumbLabel},
  };
  return output;
};

export const update = (
  rvData: number[][],
  sources: interfaces.Sources,
  tooltips: interfaces.Tooltips,
): void => {
  const computedData = computeData(rvData);
  // Update the bulk ess.
  sources.ess.bulk.line.data = computedData.ess.bulk;
  sources.ess.bulk.circle.data = computedData.ess.bulk;
  tooltips.ess.bulk.tooltips = [
    ['Total draws', '@x{0,}'],
    ['ESS', '@y{0,}'],
  ];

  // Update the tail ess.
  sources.ess.tail.line.data = computedData.ess.tail;
  sources.ess.tail.circle.data = computedData.ess.tail;
  tooltips.ess.tail.tooltips = [
    ['Total draws', '@x{0,}'],
    ['ESS', '@y{0,}'],
  ];

  // Update the rule of thumb.
  sources.ess.ruleOfThumb.line.data = computedData.ess.ruleOfThumb;
  tooltips.ess.ruleOfThumb.tooltips = [['Rule-of-thumb', '@y{0,}']];
};
