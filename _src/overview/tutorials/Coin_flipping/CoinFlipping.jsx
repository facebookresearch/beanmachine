import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure106abb80bbca402082d2b8dd9512688e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure106abb80-bbca-4020-82d2-b8dd9512688e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurecde0db57971242439c68ae3a57e7dcd5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurecde0db57-9712-4243-9c68-ae3a57e7dcd5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};