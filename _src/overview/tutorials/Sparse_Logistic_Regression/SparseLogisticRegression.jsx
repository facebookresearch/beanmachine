import React from 'react';
import { BokehFigure } from '../../../../website/src/components/Plotting.jsx';

export const SyntheticDataPlot = () => {
  const pathToData = "./assets/plot_data/synthetic-data-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const BetaMarginalsPlot = () => {
  const pathToData = "./assets/plot_data/beta-marginals-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const LambdaMarginalsPlot = () => {
  const pathToData = "./assets/plot_data/lambda-marginals-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const BetasDensityPlot = () => {
  const pathToData = "./assets/plot_data/betas-density-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const LogProbPlot = () => {
  const pathToData = "./assets/plot_data/log-prob-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const PredictedDataPlot = () => {
  const pathToData = "./assets/plot_data/predicted-data-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const DataPoint11Plot = () => {
  const pathToData = "./assets/plot_data/data-point11-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const BetaMarginalsMhPlot = () => {
  const pathToData = "./assets/plot_data/beta-marginals-mh-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const LambdaMarginalsMhPlot = () => {
  const pathToData = "./assets/plot_data/lambda-marginals-mh-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const LogProbMhPlot = () => {
  const pathToData = "./assets/plot_data/log-prob-mh-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const GermanLogProbPlot = () => {
  const pathToData = "./assets/plot_data/german-log-prob-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const GermanBetasDensityPlot = () => {
  const pathToData = "./assets/plot_data/german-betas-density-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const RocPlot = () => {
  const pathToData = "./assets/plot_data/roc-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};