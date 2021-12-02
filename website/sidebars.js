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
      'framework_topics/inference/inference',
      'framework_topics/inference/ancestral_metropolis_hastings',
      'framework_topics/inference/uniform_metropolis_hastings',
      'framework_topics/inference/random_walk',
      'framework_topics/inference/hamiltonian_monte_carlo',
      'framework_topics/inference/newtonian_monte_carlo',
      'framework_topics/programmable_inference/programmable_inference',
      'framework_topics/programmable_inference/compositional_inference',
      'framework_topics/programmable_inference/block_inference',
      'framework_topics/programmable_inference/transforms',
      'framework_topics/programmable_inference/adaptive_inference',
      'framework_topics/custom_proposers/custom_proposers',
      'framework_topics/custom_proposers/variable',
      'framework_topics/model_evaluation/diagnostics',
      'framework_topics/model_evaluation/model_comparison',
      'framework_topics/model_evaluation/posterior_predictive_checks',
      'framework_topics/development/logging',
    ],
    Advanced: ['overview/bmg/bmg', 'overview/beanstalk/beanstalk'],
    Tutorials: ['overview/tutorials/tutorials'],
    Packages: ['overview/packages/packages'],
    Contributing: ['contributing'],
    ...fbInternalOnly({FacebookIntern: ['overview/facebook/facebook']}),
    'Sitemap (formerly TOC)': ['toc'], // Once everyone is used to this being here, we'll remove the "(formerly TOC)" part
  },
};
