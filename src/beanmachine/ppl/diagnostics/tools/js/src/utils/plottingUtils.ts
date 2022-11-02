/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {Axis} from '@bokehjs/models/axes/axis';

/**
 * Update the given Bokeh Axis object with the new label string. You must use this
 * method to update axis strings using TypeScript, otherwise the ts compiler will throw
 * a type check error.
 *
 * @param {Axis} axis - The Bokeh Axis object needing a new label.
 * @param {string | null} label - The new label for the Bokeh Axis object.
 */
// eslint-disable-next-line import/prefer-default-export
export const updateAxisLabel = (axis: Axis, label: string | null): void => {
  // Type check requirement.
  if ('axis_label' in axis) {
    axis.axis_label = label;
  }
};
