import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure34a437f33efe408a80a32aa25ff3e702 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure34a437f3-3efe-408a-80a3-2aa25ff3e702.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure5bea72aa87204fb896cf6c1f81d88c17 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5bea72aa-8720-4fb8-96cf-6c1f81d88c17.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurea495423b37504721b178ea878396022d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea495423b-3750-4721-b178-ea878396022d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};