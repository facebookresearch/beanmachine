import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure4501ef12e3b94665ac2552b6ebc93264 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4501ef12-e3b9-4665-ac25-52b6ebc93264.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurea4ed58bb04da4c18899837aeadea4013 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea4ed58bb-04da-4c18-8998-37aeadea4013.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure81c0429427804f5295da5f816f8581a4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure81c04294-2780-4f52-95da-5f816f8581a4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};