import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure56197ad7b06b44d8aaa37181ed32d572 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure56197ad7-b06b-44d8-aaa3-7181ed32d572.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureb64f908ba802495f82d70dab32082cc7 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb64f908b-a802-495f-82d7-0dab32082cc7.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};