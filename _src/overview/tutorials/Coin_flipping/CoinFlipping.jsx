import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurec546af990c6148d29491b6aa631d0494 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec546af99-0c61-48d2-9491-b6aa631d0494.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure90a5545b4f134b5183eb24d7b61204c5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure90a5545b-4f13-4b51-83eb-24d7b61204c5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};