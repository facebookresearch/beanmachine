/* import {calculateHistogram} from 'compute-histogram'; */
import {linearRange, numericalSort, shape} from './array';
import {rankData, scaleToOne} from './dataTransformation';
import {mean as computeMean} from './pointStatistic';

/**
 * Compute the histogram of the given data.
 *
 * @param {number[]} data - Data to bin.
 * @param {number} [numBins] - The number of bins to use for the histogram. If none is
 *     given, then we follow ArviZ's implementation by using twice then number of bins
 *     of the Sturges formula.
 * @returns {number[][]} [TODO:description]
 */
export const calculateHistogram = (data: number[], numBins: number = 0): number[][] => {
  const sortedData = numericalSort(data);
  const numSamples = sortedData.length;
  const dataMin = Math.min(...data);
  const dataMax = Math.max(...data);
  if (numBins === 0) {
    numBins = Math.floor(Math.ceil(2 * Math.log2(numSamples)) + 1);
  }
  const binSize =
    (dataMax - dataMin) / numBins === 0 ? 1 : (dataMax - dataMin) / numBins;
  const bins = Array(numBins)
    .fill([0, 0])
    .map((_, i) => {
      return [i, 0];
    });

  for (let i = 0; i < data.length; i += 1) {
    const datum = sortedData[i];
    let binIndex = Math.floor((datum - dataMin) / binSize);
    // Subtract 1 if the value lies on the last bin.
    if (binIndex === numBins) {
      binIndex -= 1;
    }
    bins[binIndex][1] += 1;
  }
  return bins;
};

export interface RankHistogram {
  [key: string]: {
    quad: {
      left: number[];
      top: number[];
      right: number[];
      bottom: number[];
      chain: number[];
      draws: string[];
      rank: number[];
    };
    line: {x: number[]; y: number[]};
    chain: number[];
    rankMean: number[];
    mean: number[];
  };
}

/**
 * A histogram of rank data.
 *
 * @param {number[][]} data - Raw random variable data for several chains.
 * @returns {RankHistogram} A histogram of the data rankings.
 */
export const rankHistogram = (data: number[][]): RankHistogram => {
  const [numChains, numDraws] = shape(data);
  const numSamples = numChains * numDraws;
  const flatData = data.flat();

  // Calculate the rank of the data and ensure it is the same shape as the original
  // data.
  const rank = rankData(flatData);
  const rankArray = [];
  let start = Number.NaN;
  let end = Number.NaN;
  for (let i = 0; i < numChains; i += 1) {
    if (i === 0) {
      start = 0;
      end = numDraws;
    } else {
      start = end;
      end = (i + 1) * numDraws;
    }
    const chainRanks = rank.slice(start, end);
    rankArray.push(chainRanks);
    start = end;
    end = (i + 1) * numDraws;
  }

  // Calculate the number of bins needed. We will follow ArviZ and use twice the result
  // using the Sturges' formula.
  const numBins = Math.floor(Math.ceil(2 * Math.log2(numSamples)) + 1);
  const lastBinEdge = Math.max(...rank);

  // Calculate the bin edges. Since the linearRange function computes a linear spacing
  // of values between the start and end point, we need to ensure they are integer
  // values.
  let binEdges = linearRange(0, lastBinEdge, 1, true, numBins);
  binEdges = binEdges.map((value) => {
    return Math.ceil(value);
  });

  // Calculate the histograms of the rank data, and normalize it for each chain.
  const output = {} as RankHistogram;
  for (let i = 0; i < numChains; i += 1) {
    const chainIndex = i + 1;
    const chainName = `chain${chainIndex}`;
    const chainRankHistogram = calculateHistogram(rankArray[i], numBins);
    let counts = [];
    for (let j = 0; j < chainRankHistogram.length; j += 1) {
      counts.push(chainRankHistogram[j][1]);
    }
    counts = scaleToOne(counts);
    const chainCounts = counts.map((value) => {
      return value + i;
    });

    const chainRankMean = computeMean(chainCounts);
    const left = binEdges.slice(0, binEdges.length - 1);
    const right = binEdges.slice(1);
    const binLabel = [];
    for (let j = 0; j < left.length; j += 1) {
      binLabel.push(`${left[j].toLocaleString()}-${right[j].toLocaleString()}`);
    }
    const x = linearRange(0, numSamples, 1);
    const y = Array(x.length).fill(chainRankMean);
    output[chainName] = {
      quad: {
        left: left,
        top: chainCounts,
        right: right,
        bottom: Array(numBins).fill(i),
        chain: Array(left.length).fill(i + 1),
        draws: binLabel,
        rank: counts,
      },
      line: {x: x, y: y},
      chain: Array(x.length).fill(i + 1),
      rankMean: Array(x.length).fill(chainIndex - chainRankMean),
      mean: Array(x.length).fill(computeMean(counts)),
    };
  }
  return output;
};
