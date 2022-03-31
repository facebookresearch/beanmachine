import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure82f42583c9c648abbf8d500cbbcb15e3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure82f42583-c9c6-48ab-bf8d-500cbbcb15e3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure2c7c313e0c3e4dc7b66b2b390ca7f3a9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2c7c313e-0c3e-4dc7-b66b-2b390ca7f3a9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};