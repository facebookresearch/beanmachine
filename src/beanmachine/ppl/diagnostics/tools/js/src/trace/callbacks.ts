/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {Axis} from '@bokehjs/models/axes/axis';
import * as interfaces from './interfaces';
import {linearRange, shape} from '../stats/array';
import {interval as hdiInterval} from '../stats/highestDensityInterval';
import {rankHistogram} from '../stats/histogram';
import {oneD} from '../stats/marginal';
import {mean} from '../stats/pointStatistic';

const figureNames = ['marginals', 'forests', 'traces', 'ranks'];

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
 * Compute data for the trace diagnostic tool.
 *
 * @param {number[][]} data - Raw random variable data from the model for all chains.
 * @param {number} bwFactor - Multiplicative factor to be applied to the bandwidth when
 *     calculating the Kernel Density Estimate (KDE).
 * @param {number} hdiProbability - The highest density interval probability to use when
 *     calculating the HDI.
 * @returns {interfaces.Data} Data object that contains data for each figure including
 *     each chain.
 */
export const computeData = (
  data: number[][],
  bwFactor: number,
  hdiProbability: number,
): interfaces.Data => {
  const [numChains, numDraws] = shape(data);
  const output = {} as interfaces.Data;
  for (let i = 0; i < figureNames.length; i += 1) {
    const figureName = figureNames[i];
    if (figureName !== 'ranks') {
      switch (figureName) {
        case 'marginals':
          output[figureName] = {} as interfaces.MarginalDataAllChains;
          break;
        case 'forests':
          output[figureName] = {} as interfaces.ForestDataAllChains;
          break;
        case 'traces':
          output[figureName] = {} as interfaces.TraceDataAllChains;
          break;
        default:
          break;
      }
      for (let j = 0; j < numChains; j += 1) {
        const chainIndex = j + 1;
        const chainName = `chain${chainIndex}`;
        const chainData = data[j];
        const marginal = oneD(chainData, bwFactor);
        const marginalMean = mean(marginal.x);
        let hdiBounds;
        switch (figureName) {
          case 'marginals':
            output[figureName][chainName] = {} as interfaces.MarginalDataSingleChain;
            output[figureName][chainName] = {
              line: {x: marginal.x, y: marginal.y},
              chain: chainIndex,
              mean: marginalMean,
              bandwidth: marginal.bandwidth,
            };
            break;
          case 'forests':
            output[figureName][chainName] = {} as interfaces.ForestDataSingleChain;
            hdiBounds = hdiInterval(chainData, hdiProbability);
            output[figureName][chainName] = {
              line: {
                x: [hdiBounds.lowerBound, hdiBounds.upperBound],
                y: Array(2).fill(chainIndex),
              },
              circle: {x: [marginalMean], y: [chainIndex]},
              chain: chainIndex,
              mean: marginalMean,
            };
            break;
          case 'traces':
            output[figureName][chainName] = {} as interfaces.TraceDataSingleChain;
            output[figureName][chainName] = {
              line: {x: linearRange(0, numDraws - 1, 1), y: chainData},
              chain: chainIndex,
              mean: marginalMean,
            };
            break;
          default:
            break;
        }
      }
    } else if (figureName === 'ranks') {
      output[figureName] = rankHistogram(data);
    }
  }
  return output;
};

/**
 * Callback used to update the Bokeh application in the notebook.
 *
 * @param {number[][]} data - Raw random variable data from the model for all chains.
 * @param {string} rvName - The name of the random variable from the model.
 * @param {number} bwFactor - Multiplicative factor to be applied to the bandwidth when
 *     calculating the kernel density estimate.
 * @param {number} hdiProbability - The highest density interval probability to use when
 *     calculating the HDI.
 * @param {interfaces.Sources} sources - Bokeh sources used to render glyphs in the
 *     application.
 * @param {interfaces.Figures} figures - Bokeh figures shown in the application.
 * @param {interfaces.Tooltips} tooltips - Bokeh tooltips shown on the glyphs.
 */
export const update = (
  data: number[][],
  rvName: string,
  bwFactor: number,
  hdiProbability: number,
  sources: interfaces.Sources,
  figures: interfaces.Figures,
  tooltips: interfaces.Tooltips,
): void => {
  const [numChains] = shape(data);
  const computedData = computeData(data, bwFactor, hdiProbability);
  for (let i = 0; i < figureNames.length; i += 1) {
    const figureName = figureNames[i];
    const figure = figures[figureName];
    for (let j = 0; j < numChains; j += 1) {
      const chainIndex = j + 1;
      const chainName = `chain${chainIndex}`;
      const chainData = computedData[figureName][chainName];
      const source = sources[figureName][chainName];
      switch (figureName) {
        case 'marginals':
          source.line.data = {
            x: chainData.line.x,
            y: chainData.line.y,
            chain: Array(chainData.line.x.length).fill(chainData.chain),
            mean: Array(chainData.line.x.length).fill(chainData.mean),
          };
          updateAxisLabel(figure.below[0], rvName);
          tooltips[figureName][j].tooltips = [
            ['Chain', '@chain'],
            ['Mean', '@mean'],
            [rvName, '@x'],
          ];
          break;
        case 'forests':
          source.line.data = {
            x: chainData.line.x,
            y: chainData.line.y,
            chain: Array(chainData.line.x.length).fill(chainData.chain),
            mean: Array(chainData.line.x.length).fill(chainData.mean),
          };
          source.circle.data = {
            x: chainData.circle.x,
            y: chainData.circle.y,
            chain: [chainData.chain],
            mean: [chainData.mean],
          };
          updateAxisLabel(figure.below[0], rvName);
          tooltips[figureName][j].tooltips = [
            ['Chain', '@chain'],
            [rvName, '@mean'],
          ];
          break;
        case 'traces':
          source.line.data = {
            x: chainData.line.x,
            y: chainData.line.y,
            chain: Array(chainData.line.x.length).fill(chainData.chain),
            mean: Array(chainData.line.x.length).fill(chainData.mean),
          };
          updateAxisLabel(figure.left[0], rvName);
          tooltips[figureName][j].tooltips = [
            ['Chain', '@chain'],
            ['Mean', '@mean'],
            [rvName, '@y'],
          ];
          break;
        case 'ranks':
          source.line.data = {
            x: chainData.line.x,
            y: chainData.line.y,
            chain: chainData.chain,
            rankMean: chainData.rankMean,
          };
          tooltips[figureName][j].line.tooltips = [
            ['Chain', '@chain'],
            ['Rank mean', '@rankMean'],
          ];
          source.quad.data = chainData.quad;
          tooltips[figureName][j].quad.tooltips = [
            ['Chain', '@chain'],
            ['Draws', '@draws'],
            ['Rank', '@rank'],
          ];
          break;
        default:
          break;
      }
    }
  }
};
