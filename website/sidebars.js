/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */
const {fbInternalOnly} = require('internaldocs-fb-helpers');

module.exports = {
  someSidebar: {
    Overview: [
      'overview/introduction/introduction',
      'overview/quick_start/quick_start',
      'overview/modeling/modeling',
      'overview/inference/inference',
      'overview/analysis/analysis',
      'overview/installation/installation',
    ],
    Framework: [
      'framework_topics/inference/inference', // Throwing this at top, even though it's embedded below, because I believe it belongs up here (??)
      {
        // This whole section needs to be edited (heavily) and probably organized formally, rather than auto-generated
        type: 'autogenerated',
        dirName: 'framework_topics', // '.' means the current docs folder
      },
    ],
    Advanced: ['overview/bmg/bmg', 'overview/beanstalk/beanstalk'],
    Tutorials: ['overview/tutorials/tutorials'],
    API: ['overview/api/api'], // #TODO: Brian Johnson will populate this!
    Packages: ['overview/packages/packages'],
    Contributing: ['contributing'],
    ...fbInternalOnly({FacebookIntern: ['overview/facebook/facebook']}),
    'Sitemap (formerly TOC)': ['toc'], // Once everyone is used to this being here, we'll remove the "(formerly TOC)" part
  },
};
