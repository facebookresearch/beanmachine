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
        {
          'Inference Methods': [
            'framework_topics/inference/inference',
            'framework_topics/inference/ancestral_metropolis_hastings',
            'framework_topics/inference/random_walk',
            'framework_topics/inference/uniform_metropolis_hastings',
            'framework_topics/inference/hamiltonian_monte_carlo',
            'framework_topics/inference/newtonian_monte_carlo',
          ],
          'Programmable Inference': [
            'framework_topics/programmable_inference/programmable_inference',
            'framework_topics/programmable_inference/transforms',
            'framework_topics/programmable_inference/block_inference',
            'framework_topics/programmable_inference/compositional_inference',
            'framework_topics/programmable_inference/adaptive_inference',
          ],
          'Custom Proposers': [
            'framework_topics/custom_proposers/variable',
            'framework_topics/custom_proposers/custom_proposers',
          ],
          'Model Evaluation': [
            'framework_topics/model_evaluation/diagnostics',
            'framework_topics/model_evaluation/posterior_predictive_checks',
            'framework_topics/model_evaluation/model_comparison',
          ],
          Development: ['framework_topics/development/logging'],
        },
      ],
      Advanced: [
        'overview/beanstalk/beanstalk',
        'overview/bean_machine_advantages/bean_machine_advantages',
        ],
    },
    'overview/tutorials/tutorials',
    // TODO(sepehrakhavan): Re-display this once we have at least one package present.
    // {
    //   Packages: ['overview/packages/packages'],
    // },
    'contributing',
    {
      ...fbInternalOnly({FacebookIntern: ['overview/facebook/facebook']}),
    },
  ],
};
