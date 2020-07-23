/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const features = [
  {
    title: <>Declarative modeling</>,
    // imageUrl: 'img/undraw_docusaurus_tree.svg',
    description: (
      <>
        Clean, intuitive syntax that lets you focus on the model
        and leave performance to the framework.
      </>
    ),
  },
  {
    title: <>Programmable inference</>,
    // imageUrl: 'img/undraw_docusaurus_tree.svg',
    description: (
      <>
        Mix-and-match inference methods, proposers, and
        inference strategies to achieve maximum efficiency.
      </>
    ),
  },
  {
    title: <>Powered by React</>,
    // imageUrl: 'img/undraw_docusaurus_react.svg',
    description: (
      <>
        Leverage native GPU and autograd support and
        integrate seamlessly with the PyTorch ecosystem.
      </>
    ),
  },
];


function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx('col col--4', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="BeanMachine">
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
        <img className={styles.heroLogo} src='img/beanmachine.svg' alt="BeanMachine Logo." width="100"/>
          <img className="imgUrl">{siteConfig.imgUrl}</img>
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={clsx(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('docs/overview/introduction/introduction')}>
              Get Started
            </Link>
          </div>
        </div>
      </header>
      <main>

        {features && features.length > 0 && (
          <section className={styles.features}>
            <div className='container'>
              <div className='row'>
                {features.map(({title, imageUrl, description}) => (
                  <Feature
                    key={title}
                    title={title}
                    imageUrl={imageUrl}
                    description={description}
                  />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;
