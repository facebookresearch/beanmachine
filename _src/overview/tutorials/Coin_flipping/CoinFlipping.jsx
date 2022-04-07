import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure785e46188801407f9b4399e964737c97 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure785e4618-8801-407f-9b43-99e964737c97.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure8f0c0f4310574f60a566b3c294996d33 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8f0c0f43-1057-4f60-a566-b3c294996d33.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};