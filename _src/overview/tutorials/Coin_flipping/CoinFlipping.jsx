import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureb438676f384a49239667707415dc81a9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb438676f-384a-4923-9667-707415dc81a9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure98a383f311da40818eb5a67c60a51439 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure98a383f3-11da-4081-8eb5-a67c60a51439.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};