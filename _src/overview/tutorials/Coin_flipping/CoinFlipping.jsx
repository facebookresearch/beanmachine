import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure56b9d96ae6eb4bd59b18ab2d7d386f4a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure56b9d96a-e6eb-4bd5-9b18-ab2d7d386f4a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureb6a8b799dcdb40eabee9be9244f89bc0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb6a8b799-dcdb-40ea-bee9-be9244f89bc0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};