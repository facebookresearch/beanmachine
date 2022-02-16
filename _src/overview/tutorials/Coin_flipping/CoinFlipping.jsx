import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguref80ad6368f35485dac99f1c52d575be8 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref80ad636-8f35-485d-ac99-f1c52d575be8.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurea62798a9ccf845c482a8f3a7cc7ead41 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea62798a9-ccf8-45c4-82a8-f3a7cc7ead41.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};