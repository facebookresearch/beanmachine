import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureba1a82ccc85c40f8a6074cd1bb0552eb = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureba1a82cc-c85c-40f8-a607-4cd1bb0552eb.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure6f4826c620c147be823bba4e8e31d9e4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6f4826c6-20c1-47be-823b-ba4e8e31d9e4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};