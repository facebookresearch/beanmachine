import React from 'react';
import { BokehFigure } from '../../../../website/src/components/Plotting.jsx';

export const ValueCountsPlot = () => {
  const pathToData = "./assets/plot_data/value-counts-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const FirstModelPriorPc = () => {
  const pathToData = "./assets/plot_data/first-model-prior-pc.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const PriorPostPlots = () => {
  const pathToData = "./assets/plot_data/prior-post-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SecModelPriorPlot = () => {
  const pathToData = "./assets/plot_data/sec-model-prior-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SecPriorPostPlot = () => {
  const pathToData = "./assets/plot_data/sec-prior-post-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const PThetaPlots = () => {
  const pathToData = "./assets/plot_data/p-theta-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SecondModelDiagPlot = () => {
  const pathToData = "./assets/plot_data/second-model-diag-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const SecThrdModelPostPlot = () => {
  const pathToData = "./assets/plot_data/sec-thrd-model-post-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const ThrdPThetaPlots = () => {
  const pathToData = "./assets/plot_data/thrd-p-theta-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const DiagPlots = () => {
  const pathToData = "./assets/plot_data/diag-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const FrstScndThrdPlots = () => {
  const pathToData = "./assets/plot_data/frst-scnd-thrd-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const ADPlots = () => {
  const pathToData = "./assets/plot_data/a-d-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const ThPBPlots = () => {
  const pathToData = "./assets/plot_data/th-p-b-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};