import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure0a03b6278ac4479b9b0a75a4869c83a0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure0a03b627-8ac4-479b-9b0a-75a4869c83a0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigured410a682bca74e3e8214db077010d35e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured410a682-bca7-4e3e-8214-db077010d35e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};