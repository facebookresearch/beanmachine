/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as marginal2d from './callbacks';

// The CustomJS methods used by Bokeh require us to make the JavaScript available in the
// browser, which is done by defining it below.
(window as any).marginal2d = marginal2d;
