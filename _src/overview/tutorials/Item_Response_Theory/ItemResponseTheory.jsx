import React from 'react';
import { BokehFigure } from '../../../../website/src/components/Plotting.jsx';

export const FoulTypesPlot = () => {
  const pathToData = "./assets/plot_data/foul-types-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const CallTypePlot = () => {
  const pathToData = "./assets/plot_data/call-type-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const FoulFreqPlot = () => {
  const pathToData = "./assets/plot_data/foul-freq-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const BasicModelPosteriorPlot = () => {
  const pathToData = "./assets/plot_data/basic-model-posterior-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const StrTracePlot = () => {
  const pathToData = "./assets/plot_data/str-trace-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const BasicAutocorrPlot = () => {
  const pathToData = "./assets/plot_data/basic-autocorr-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const BasicResidPlot = () => {
  const pathToData = "./assets/plot_data/basic-resid-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const TrailingTeamCommittingPlot = () => {
  const pathToData = "./assets/plot_data/trailing-team-committing-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const TrailingPossPlot = () => {
  const pathToData = "./assets/plot_data/trailing-poss-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const TypePlot = () => {
  const pathToData = "./assets/plot_data/type-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const PossAutocorrPlot = () => {
  const pathToData = "./assets/plot_data/poss-autocorr-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const PossResidPlot = () => {
  const pathToData = "./assets/plot_data/poss-resid-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};

export const IrtResidPlot = () => {
  const pathToData = "./assets/plot_data/irt-resid-plot.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};