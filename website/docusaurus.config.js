/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const math = require('remark-math');
const katex = require('rehype-katex');
const {fbContent} = require('internaldocs-fb-helpers');

module.exports = {
  title: 'Bean Machine',
  tagline:
    'A universal probabilistic programming language to enable fast and accurate Bayesian analysis',
  url: 'https://home.fburl.com/ppl', // Change to path for release.
  baseUrl: '/', // for devserver preview use '/~brianjo/beanmachine/'
  favicon: 'img/favicon.ico',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  organizationName: 'facebook', // Usually your GitHub org/user name.
  projectName: 'beanmachine', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'Bean Machine',
      logo: {
        alt: 'Bean Machine Logo',
        src: 'img/beanmachine.svg',
      },
      items: [
        {
          to: 'users',
          activeBasePath: 'users',
          label: 'Users',
          position: 'left',
        },
        {
          to: 'api',
          activeBasePath: 'api',
          label: 'API',
          position: 'left',
        },
        // {href: '/api/index.html', label: 'API', position: 'left'},
        // {to: 'blog', label: 'Blog', position: 'left'},
        // Please keep GitHub link to the right for consistency.
        {
          href: 'https://github.com/facebook/docusaurus',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      /* Commenting this section out while the fix a DS2 issue.
      links: [
        {
          title: 'Learn',
          items: [
            {
              label: 'Quick Start',
              to: '/docs/overview/quick_start/quick_start/',
            },
            {
              label: 'Modeling',
              to: '/docs/overview/modeling/modeling/',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/docusaurus',
            },
            {
              label: 'Twitter',
              href: 'https://twitter.com/docusaurus',
            },
            {
              label: 'Discord',
              href: 'https://discordapp.com/invite/docusaurus',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: 'blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/facebook/docusaurus',
            },
          ],
        },
        {
          title: 'Legal',
          // Please do not remove the privacy and terms, it's a legal requirement.
          items: [
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
          ],
        },
      ],
      */

      logo: {
        alt: 'Bean Machine Logo',
        src: 'img/beanmachine.svg',
        href: 'https://opensource.facebook.com',
      },
      // Please do not remove the credits, help to publicize Docusaurus :)
      copyright: `Copyright &#169; ${new Date().getFullYear()} Facebook, Inc. Built with Docusaurus.`,
    },
  },
  stylesheets: ['https://cdn.jsdelivr.net/npm/katex@0.11.0/dist/katex.min.css'],
  presets: [
    [
      require.resolve('docusaurus-plugin-internaldocs-fb/docusaurus-preset'),
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: '/',
          // Please change this to your repo.
          editUrl: fbContent({
            internal: 'https://www.internalfb.com/intern/diffusion/FBS/browse/master/fbcode/beanmachine/website/',
            external: 'https://github.com/facebook/docusaurus/edit/master/website/'
          }),
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
