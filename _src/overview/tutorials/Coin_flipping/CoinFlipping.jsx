import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure5dbf412f605944d08dd90b7ca0bc0888 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5dbf412f-6059-44d0-8dd9-0b7ca0bc0888.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguref7d97d16eb8c4c47a5da016ceeb5642e = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref7d97d16-eb8c-4c47-a5da-016ceeb5642e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};