import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure1d37a5419dbd4f3c9fee7fabdc805b89 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure1d37a541-9dbd-4f3c-9fee-7fabdc805b89.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure38361d0e5ad6445e803d6656ab3d2592 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure38361d0e-5ad6-445e-803d-6656ab3d2592.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};