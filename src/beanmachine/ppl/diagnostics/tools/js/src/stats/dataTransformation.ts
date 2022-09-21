// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

export const scaleBy = (data: number[], scaleFactor: number): number[] => {
  const scaledData = data.map((datum) => {
    return datum / scaleFactor;
  });
  return scaledData;
};

export const scaleToOne = (data: number[]): number[] => {
  const scaleFactor = Math.max(...data);
  return scaleBy(data, scaleFactor);
};
