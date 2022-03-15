import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea6a5bed84f9f4ff5a835eadc245f186f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea6a5bed8-4f9f-4ff5-a835-eadc245f186f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure64b36212d09e489e9b390094e30d9b85 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure64b36212-d09e-489e-9b39-0094e30d9b85.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};