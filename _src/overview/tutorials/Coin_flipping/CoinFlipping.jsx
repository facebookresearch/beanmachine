import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure9d4b8a7e0a29473abe04afcffa802090 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9d4b8a7e-0a29-473a-be04-afcffa802090.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure25d8a50c3c594f35b09c1f04c327c9cb = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure25d8a50c-3c59-4f35-b09c-1f04c327c9cb.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};