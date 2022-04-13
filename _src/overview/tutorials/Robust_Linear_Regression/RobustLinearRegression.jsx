import React from 'react';
import { BokehFigure } from '../../../../website/src/components/Plotting.jsx';

export const GammaPriorPlot = () => {
  const pathToData = "./assets/plot_data/gamma-prior-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SyntheticCleanDataPlot = () => {
  const pathToData = "./assets/plot_data/synthetic-clean-data-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SyntheticDataPlot = () => {
  const pathToData = "./assets/plot_data/synthetic-data-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const AlphaBetaJointPlot = () => {
  const pathToData = "./assets/plot_data/alpha-beta-joint-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SigmaPlot = () => {
  const pathToData = "./assets/plot_data/sigma-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const TestDataMhPlot = () => {
  const pathToData = "./assets/plot_data/test-data-mh-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SamplesDiagnosticPlots = () => {
  const pathToData = "./assets/plot_data/samples-diagnostic-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const PredictionPlot = () => {
  const pathToData = "./assets/plot_data/prediction-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};