import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurecf056f202c3048afa056c0b069d56ad1 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurecf056f20-2c30-48af-a056-c0b069d56ad1.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure6c24dce3fb3a4496ad874396d87f0924 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6c24dce3-fb3a-4496-ad87-4396d87f0924.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};