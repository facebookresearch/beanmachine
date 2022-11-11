/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const path = require('path');

module.exports = {
  entry: {
    effective_sample_size: './src/ess/index.ts',
    marginal1d: './src/marginal1d/index.ts',
    trace: './src/trace/index.ts',
  },
  output: {
    filename: '[name].js',
    path: path.resolve(__dirname, 'dist'),
  },
  mode: 'production',
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js'],
    alias: {
      '@bokehjs': path.resolve(__dirname, './node_modules/@bokeh/bokehjs/build/js/lib'),
    },
  },
};
