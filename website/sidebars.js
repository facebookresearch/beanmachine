/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

var apiSidebar = require('./api.sidebar.js')

module.exports = {
  usersSidebar: {
    Documentation: [
      'users/toc',
      'users/overview/introduction/introduction',
      'users/overview/quick_start/quick_start',
      'users/overview/modeling/modeling',
      'users/overview/inference/inference',
      'users/overview/analysis/analysis',
      'users/overview/beanstalk/beanstalk',
      'users/contributing']
  },
  apiSidebar: apiSidebar,
};
