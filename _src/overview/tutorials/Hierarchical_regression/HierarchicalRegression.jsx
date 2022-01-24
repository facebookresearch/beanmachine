import React from 'react';
import { BokehFigure } from '../../../../website/src/components/Plotting.jsx';

export const LogPlotComparison = () => {
  const pathToData = "./assets/plot_data/log-plot-comparison.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const FloorPlot = () => {
  const pathToData = "./assets/plot_data/floor-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SampleOfPriors = () => {
  const pathToData = "./assets/plot_data/sample-of-priors.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const QueryEss = () => {
  const pathToData = "./assets/plot_data/query-ess.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SampleCountiesEss = () => {
  const pathToData = "./assets/plot_data/sample-counties-ess.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const TraceRanks = () => {
  const pathToData = "./assets/plot_data/trace-ranks.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const BetaGamma = () => {
  const pathToData = "./assets/plot_data/beta-gamma.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const CountyPosteriors = () => {
  const pathToData = "./assets/plot_data/county-posteriors.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const Uranium = () => {
  const pathToData = "./assets/plot_data/uranium.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};