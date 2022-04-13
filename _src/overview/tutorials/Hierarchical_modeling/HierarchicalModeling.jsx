import React from 'react';
import { BokehFigure } from '../../../../website/src/components/Plotting.jsx';

export const CurrentHits = () => {
  const pathToData = "./assets/plot_data/current-hits.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const CompletePoolingPriors = () => {
  const pathToData = "./assets/plot_data/complete-pooling-priors.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const CompletePoolingEssPlot = () => {
  const pathToData = "./assets/plot_data/complete-pooling-ess-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const CompletePoolingDiagnosticsPlot = () => {
  const pathToData = "./assets/plot_data/complete-pooling-diagnostics-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const CompletePoolingPhiPosterior = () => {
  const pathToData = "./assets/plot_data/complete-pooling-phi-posterior.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const CompletePoolingModelPlot = () => {
  const pathToData = "./assets/plot_data/complete-pooling-model-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const NoPoolingDiagPlots = () => {
  const pathToData = "./assets/plot_data/no-pooling-diag-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const CompleteNoPlot = () => {
  const pathToData = "./assets/plot_data/complete-no-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const ParetoPlot = () => {
  const pathToData = "./assets/plot_data/pareto-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const PartialPoolingDiagPlots = () => {
  const pathToData = "./assets/plot_data/partial-pooling-diag-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const ModelPlots = () => {
  const pathToData = "./assets/plot_data/model-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const ShrinkagePlot = () => {
  const pathToData = "./assets/plot_data/shrinkage-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};