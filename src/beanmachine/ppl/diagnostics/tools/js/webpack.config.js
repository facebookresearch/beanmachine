/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const path = require('path');

module.exports = {
  entry: {
    marginal1d: './src/marginal1d/index.ts',
    trace: './src/trace/index.ts',
  },
  output: {
    filename: '[name].js',
    path: path.resolve(__dirname, 'dist'),
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules\/(?!(@bokeh\/bokehjs\/build\/js\/lib)\/).*/,
      },
    ],
  },
  target: 'web',
  mode: 'production',
  resolve: {
    extensions: ['.ts', '.js'],
    modules: ['./stats', './interfaces', './types', './node_modules'],
    alias: {
      'fast-kde/src/density1d': path.resolve(
        __dirname,
        'node_modules/fast-kde/src/density1d.js',
      ),
      '@bokehjs/models/ranges/range1d': path.resolve(
        __dirname,
        'node_modules/@bokeh/bokehjs/build/js/lib/models/ranges/range1d.js',
      ),
    },
  },
  optimization: {
    minimize: false,
  },
};
