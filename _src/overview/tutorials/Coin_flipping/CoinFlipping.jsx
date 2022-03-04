import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguree725f0d9a2a449d087569e7cdcfcae0d = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguree725f0d9-a2a4-49d0-8756-9e7cdcfcae0d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec7f96058909c402b9c41816e419bfca5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec7f96058-909c-402b-9c41-816e419bfca5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};