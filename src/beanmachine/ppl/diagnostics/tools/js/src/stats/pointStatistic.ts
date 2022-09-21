// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

export const sum = (data: number[]): number => {
  return data.reduce((previousValue, currentValue) => {
    return previousValue + currentValue;
  });
};

export const mean = (data: number[]): number => {
  const dataSum = sum(data);
  return dataSum / data.length;
};
