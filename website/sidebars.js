/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const {fbInternalOnly, fbContent} = require('internaldocs-fb-helpers');
const tutorials = () => {
  const allTutorialMetadata = require('./tutorials.json');
  const tutorialPaths = ['overview/tutorials/tutorials'];
  for (var key in allTutorialMetadata) {
    tutorialPaths.push(allTutorialMetadata[key]['path']);
  }
  return tutorialPaths;
};

module.exports = {
  someSidebar: [
    {
      Overview: [
        'overview/why_bean_machine/why_bean_machine',
        'overview/quick_start/quick_start',
        'overview/modeling/modeling',
        'overview/inference/inference',
        'overview/analysis/analysis',
        'overview/installation/installation',
      ],
      Framework: [
        'framework_topics/world',
        {
          'Inference Methods': [
            'framework_topics/inference/inference',
            'framework_topics/inference/ancestral_metropolis_hastings',
            'framework_topics/inference/random_walk',
            'framework_topics/inference/uniform_metropolis_hastings',
            'framework_topics/inference/hamiltonian_monte_carlo',
            'framework_topics/inference/no_u_turn_sampler',
            'framework_topics/inference/newtonian_monte_carlo',
          ],
          'Custom Inference': [
            'framework_topics/custom_inference/programmable_inference',
            'framework_topics/custom_inference/custom_proposers',
            // Todo, stale and not fully supported
            // 'framework_topics/custom_inference/transforms',
            'framework_topics/custom_inference/compositional_inference',
            'framework_topics/custom_inference/block_inference',
          ],
          'Model Evaluation': [
            'framework_topics/model_evaluation/diagnostics',
            // TODO(rodrigobraz) - rewrite for clarity and completeness and check if simulate API has changed.
            // 'framework_topics/model_evaluation/posterior_predictive_checks',
            // 'framework_topics/model_evaluation/model_comparison',
          ],
          Development: ['framework_topics/development/logging'],
        },
      ],
      Advanced: ['overview/beanstalk/beanstalk'],
      Tutorials: fbContent({
        internal: ['overview/tutorials/tutorials'],
        external: tutorials(),
      }),
      "API Documentation": [
        require("../docs/api/sidebar.json")
      ],
    },
    // TODO(sepehrakhavan): Re-display this once we have at least one package present.
    // {
    //   Packages: ['overview/packages/packages'],
    // },
    {
      ...fbInternalOnly({FacebookIntern: ['overview/facebook/facebook']}),
    },
  ],
};
