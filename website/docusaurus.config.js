/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const remarkMath = require('remark-math');
const rehypeKatex = require('rehype-katex');
const {fbContent} = require('internaldocs-fb-helpers');

module.exports = {
  title: 'Bean Machine',
  tagline:
    'A universal probabilistic programming language to enable fast and accurate Bayesian analysis',
  url: 'https://beanmachine.org', // Change to path for release.
  baseUrl: '/', // for devserver preview use '/~brianjo/beanmachine/'
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  trailingSlash: true,
  favicon: 'img/favicon.ico',
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
          to: 'docs/overview/why_bean_machine',
          // activeBasePath: '../',
          label: 'Docs',
          position: 'left',
        },
        {
          to: '/docs/tutorials',
          label: 'Tutorials',
          position: 'left',
        },
        {
          href: 'pathname:///api/index.html',
          label: 'API',
          position: 'left',
        },
        // {to: 'blog', label: 'Blog', position: 'left'},
        // Please keep GitHub link to the right for consistency.
        {
          href: 'https://github.com/facebookresearch/beanmachine',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
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

      logo: {
        alt: 'Bean Machine Logo',
        src: 'img/beanmachine.svg',
        href: 'https://opensource.facebook.com',
      },
      // Please do not remove the credits, help to publicize Docusaurus :)
      copyright: `Copyright &#169; ${new Date().getFullYear()} Meta Platforms, Inc. Built with Docusaurus.`,
    },
  },
  stylesheets: ['https://cdn.jsdelivr.net/npm/katex@0.11.0/dist/katex.min.css'],
  scripts: ['https://cdn.bokeh.org/bokeh/release/bokeh-2.4.2.min.js'],
  presets: [
    [
      require.resolve('docusaurus-plugin-internaldocs-fb/docusaurus-preset'),
      {
        docs: {
          // It is recommended to set document id as docs home page (`docs/` path).
          path: '../docs/',
          // homePageId: '/docs/toc',
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: fbContent({
            internal:
              'https://www.internalfb.com/intern/diffusion/FBS/browse/master/fbcode/beanmachine/website/',
            external:
              'https://github.com/facebookresearch/beanmachine/edit/main/website/',
          }),
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl: fbContent({
            internal:
              'https://www.internalfb.com/intern/diffusion/FBS/browse/master/fbcode/beanmachine/website/blog/',
            external:
              'https://github.com/facebookresearch/beanmachine/edit/main/website/blog/',
          }),
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        googleAnalytics: {
           trackingID: 'UA-44373548-47',
        },
      },
    ],
  ],
};
