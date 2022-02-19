import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure16d7d8e0539c4dc6b93c645136a1320e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure16d7d8e0-539c-4dc6-b93c-645136a1320e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure2dd3ba3543ce4bf4b1a196d00bb9cae7 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2dd3ba35-43ce-4bf4-b1a1-96d00bb9cae7.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};