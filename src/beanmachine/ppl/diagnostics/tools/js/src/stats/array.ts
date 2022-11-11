/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Cumulative sum of the given data.
 *
 * @param {number[]} data - Any array of data.
 * @returns {number[]} The cumulative sum of the given data.
 */
export const cumulativeSum = (data: number[]): number[] => {
  // eslint-disable-next-line arrow-body-style
  const cumulativeSumMap = ((sum: number) => (value: number) => {
    // eslint-disable-next-line no-return-assign
    return (sum += value);
  })(0);

  const cSum = data.map(cumulativeSumMap);
  const normalizationFactor = Math.max(...cSum);
  const cumSum = [];
  for (let i = 0; i < cSum.length; i += 1) {
    cumSum.push(cSum[i] / normalizationFactor);
  }
  return cumSum;
};

/**
 * Numerically sort the given array of numbers.
 *
 * @param {number[]} data - The array of numbers to sort.
 * @returns {number[]} The sorted array of numbers.
 */
export const numericalSort = (data: number[]): number[] => {
  const dataCopy = data.slice(0);
  return dataCopy.sort((a, b) => {
    return a < b ? -1 : a > b ? 1 : 0;
  });
};

/**
 * Determine the shape of the given array.
 *
 * @param {any[]} data - Any array of data.
 * @returns {number[]} The shape of the data as an array.
 */
export const shape = (data: any[]): number[] => {
  // From https://stackoverflow.com/questions/10237615/get-size-of-dimensions-in-array
  const computeShape = (array: any[]): any[] => {
    return array.length ? [...[array.length], ...computeShape(array[0])] : [];
  };
  const arrayShape = computeShape(data);
  // Remove the empty array that will exist at the end of the shape array, since it is
  // the returned "else" value from above.
  const dataShape = [];
  for (let i = 0; i < arrayShape.length; i += 1) {
    if (!Array.isArray(arrayShape[i])) {
      dataShape.push(arrayShape[i]);
    }
  }
  return dataShape;
};

/**
 * Create an array that starts and stops with the given number of steps.
 *
 * @param {number} start - Where to start the array from.
 * @param {number} stop - Where to stop the array.
 * @param {number} [step] - The step size to take.
 * @param {boolean} [closed] - Flag used to return a closed array or not.
 * @param {null | number} [size] - If not null, then will return an array with the given
 *     size.
 * @returns {number[]} An array that is linearly spaced between the start and stop
 *     values.
 */
export const linearRange = (
  start: number,
  stop: number,
  step: number = 1,
  closed: boolean = true,
  size: null | number = null,
): number[] => {
  if (size !== null) {
    step = (stop - start) / size;
  }
  let len = (stop - start) / step + 1;
  if (!closed) {
    len = (stop - start - step) / step + 1;
  }
  return Array.from({length: len}, (_, i) => {
    return start + i * step;
  });
};

/**
 * Return the indices that would sort the array. Follows NumPy's implementation.
 *
 * @param {number[]} data - The data to sort.
 * @returns {number[]} An array of indices that would sort the original array.
 */
export const argSort = (data: number[]): number[] => {
  const dataCopy = data.slice(0);
  return dataCopy
    .map((value, index) => {
      return [value, index];
    })
    .sort((a, b) => {
      return a[0] - b[0];
    })
    .map((value) => {
      return value[1];
    });
};

/**
 * Count the number of time a value appears in an array.
 *
 * @param {number[]} data - The numeric array to count objects for.
 * @returns {{[key: string]: number}} An object that contains the keys as the items in
 *     the original array, and values that are counts of the key.
 */
export const valueCounts = (data: number[]): {[key: string]: number} => {
  const counts: {[key: string]: number} = {};
  for (let i = 0; i < data.length; i += 1) {
    counts[data[i]] = (counts[data[i]] || 0) + 1;
  }
  return counts;
};

/**
 * Create an empty 2D array with the given dimensions.
 *
 * @param {number} numRows - The number of rows to create the empty array with.
 * @param {number} numColumns - The number of columns to create the empty array with.
 * @returns {number[][]} The empty array.
 */
export const createEmpty2dArray = (numRows: number, numColumns: number): number[][] => {
  return [...Array(numRows)].map(() => {
    return [...Array(numColumns)];
  });
};

/**
 * Computes empirical quantiles for the given data. Follows the implementation in SciPy
 * closely. See
 * https://github.com/scipy/scipy/blob/2d49ca49099b498bf847c248c0878b7f0037c0b2/scipy/stats/_mstats_basic.py#L3058
 * for a more detailed description.
 *
 * @param {number[][]} data - Input data.
 * @param {number} probability - The quantile to compute.
 * @param {number} [alphap] - Plotting positions parameter.
 * @param {number} [betap] - Plotting positions parameter.
 * @returns {number} The calculated quantile.
 */
export const quantile = (
  data: number[][],
  probability: number,
  alphap: number = 0.4,
  betap: number = 1.0,
): number => {
  // scipy.mquantiles
  const flattenedData = data.flat();
  const sortedData = numericalSort(flattenedData);
  const n = sortedData.length;
  const m = alphap + probability * (1 - alphap - betap);

  // scipy.mquantiles._quantiles1D
  let quantile1d: number;
  if (n === 0) {
    quantile1d = Number.NaN;
  } else if (n === 1) {
    [quantile1d] = sortedData;
  } else {
    const aleph = n * probability + m;
    const k = Math.floor(aleph);
    const gamma = aleph - k;
    quantile1d = (1.0 - gamma) * sortedData[k - 1] + gamma * sortedData[k];
  }
  return quantile1d;
};

/**
 * Split the given array into two arrays. This implementation follows ArviZ closely.
 *
 * @param {number[]} data - 1D input array of data.
 * @returns {number[][]} The data split into two arrays.
 */
export const splitArray = (data: number[]): number[][] => {
  const numDraws = data.length;
  const halfIndex = Math.floor(numDraws / 2);
  const firstHalfOfDraws = data.slice(0, halfIndex);
  let lastHalfOfDraws = data.slice(halfIndex, numDraws);
  // NOTE: This follows the original pattern in ArviZ.
  if (firstHalfOfDraws.length !== lastHalfOfDraws.length) {
    lastHalfOfDraws = data.slice(halfIndex + 1, numDraws);
  }
  return [firstHalfOfDraws, lastHalfOfDraws];
};

/**
 * Stack the given arrays in the same way ArviZ does.
 *
 * @param {number[][][]} splitChainData - Chain data that has been split.
 * @returns {number[][]} Data that has had its dimensionality reduced from three
 *     dimensions to two.
 */
export const stackArrays = (splitChainData: number[][][]): number[][] => {
  // Reduce the given data into a 2D array from a 3D one.
  const stackedSplitChains = splitChainData.reduce((previousValue, currentValue) => {
    return previousValue.concat(currentValue);
  });
  const evenSplitChainEntries = stackedSplitChains.filter((_, index) => {
    return index % 2 === 0;
  });
  const oddSplitChainEntries = stackedSplitChains.filter((_, index) => {
    return index % 2 !== 0;
  });
  const stackedData = evenSplitChainEntries.concat(oddSplitChainEntries);
  return stackedData;
};
