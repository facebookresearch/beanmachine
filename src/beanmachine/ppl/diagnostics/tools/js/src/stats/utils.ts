// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

const left_edge_index = (point: number, intervals: number[]) => {
  const len = intervals.length;
  if (point < intervals[0]) {
    return -1;
  }
  if (point > intervals[len - 1]) {
    return intervals.length;
  }
  let leftEdgeIndex = 0;
  let rightEdgeIndex = len - 1;
  while (rightEdgeIndex - leftEdgeIndex !== 1) {
    const indexOfNumberToCompare =
      leftEdgeIndex + Math.floor((rightEdgeIndex - leftEdgeIndex) / 2);
    if (point >= intervals[indexOfNumberToCompare])
      leftEdgeIndex = indexOfNumberToCompare;
    else rightEdgeIndex = indexOfNumberToCompare;
  }
  return leftEdgeIndex;
};

const lerp = (x: number, x0: number, y0: number, x1: number, y1: number): number => {
  const slope = (y1 - y0) / (x1 - x0);
  let res = slope * (x - x0) + y0;
  if (!Number.isFinite(res)) {
    res = slope * (x - x1) + y1;
    if (!Number.isFinite(res) && y0 === y1) res = y0;
  }
  return res;
};

const interpolate = (points: number[], x: number[], y: number[]): number[] => {
  // Implementation ported from np.interp
  const n = points.length;
  const results = new Array(n);
  for (let i = 0; i < n; i += 1) {
    const point = points[i];
    if (Number.isNaN(point)) {
      results[i] = point;
      // eslint-disable-next-line no-continue
      continue;
    }
    const index = left_edge_index(point, x);
    if (index === -1) {
      // eslint-disable-next-line prefer-destructuring
      results[i] = y[0];
    } else if (index === x.length) {
      results[i] = y[y.length - 1];
    } else if (index === x.length - 1 || x[index] === point) {
      results[i] = y[index];
    } else {
      const x0 = x[index];
      const y0 = y[index];
      const x1 = x[index + 1];
      const y1 = y[index + 1];
      results[i] = lerp(point, x0, y0, x1, y1);
    }
  }
  return results;
};

// eslint-disable-next-line import/prefer-default-export
export const interpolatePoints = ({
  x,
  y,
  points,
}: {
  x: number[];
  y: number[];
  points: number[];
}): number[] => {
  const output = [];
  for (let i = 0; i < points.length; i += 1) {
    const interpolatedPoint = Array.from(interpolate([points[i]], x, y));
    output.push(interpolatedPoint[0]);
  }
  return output;
};
