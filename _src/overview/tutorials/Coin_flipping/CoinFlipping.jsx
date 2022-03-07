import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea7dd7e45bbd5411f96d8e292c69195c9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea7dd7e45-bbd5-411f-96d8-e292c69195c9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguref8219ecb12f345f7b4f0ec8f1f0c66cf = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref8219ecb-12f3-45f7-b4f0-ec8f1f0c66cf.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};