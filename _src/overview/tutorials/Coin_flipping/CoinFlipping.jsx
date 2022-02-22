import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure1b55e97bb0aa42d68d57778dcbf1d8ab = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure1b55e97b-b0aa-42d6-8d57-778dcbf1d8ab.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure4e1cb230f5f34db388666696597826a5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4e1cb230-f5f3-4db3-8866-6696597826a5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};