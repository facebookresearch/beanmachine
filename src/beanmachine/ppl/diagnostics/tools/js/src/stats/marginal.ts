// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {density1d} from 'fast-kde/src/density1d';
import {scaleToOne} from './dataTransformation';

// eslint-disable-next-line import/prefer-default-export
export const oneD = (
  data: number[],
  bwFactor: number,
): {x: number[]; y: number[]; bandwidth: number} => {
  // Calculate the 1D KDE.
  const kde1d = density1d(data, {
    adjust: bwFactor,
    pad: 3,
    bins: 512,
  });
  // Extract the data points from the 1D KDE calculation.
  const points: {x: number; y: number}[] = Array.from(kde1d);
  const x = [];
  const y = [];
  for (let i = 0; i < points.length; i += 1) {
    x.push(points[i].x);
    y.push(points[i].y);
  }
  return {
    x: x,
    y: scaleToOne(y),
    bandwidth: kde1d.bandwidth(),
  };
};
