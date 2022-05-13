import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure6e33080f1fa447cdace6dcc12be0770e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6e33080f-1fa4-47cd-ace6-dcc12be0770e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure71efc03e0672424eb874100cce8db40a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure71efc03e-0672-424e-b874-100cce8db40a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure8c74e62bfc114bb79b74f18bb6616887 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8c74e62b-fc11-4bb7-9b74-f18bb6616887.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};