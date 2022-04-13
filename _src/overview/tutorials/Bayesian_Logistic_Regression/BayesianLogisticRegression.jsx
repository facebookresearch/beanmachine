import React from 'react';
import { BokehFigure } from '../../../../website/src/components/Plotting.jsx';

export const SyntheticDataPlot = () => {
  const pathToData = "./assets/plot_data/synthetic-data-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SyntheticDataWithCategoriesPlot = () => {
  const pathToData = "./assets/plot_data/synthetic-data-with-categories-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SlopeHistPlot = () => {
  const pathToData = "./assets/plot_data/slope-hist-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const InterceptHistPlot = () => {
  const pathToData = "./assets/plot_data/intercept-hist-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const RandomlySelectedLinesPlot = () => {
  const pathToData = "./assets/plot_data/randomly-selected-lines-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};