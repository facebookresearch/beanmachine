import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureaf3409d4406a43219bdab5cf338fd671 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureaf3409d4-406a-4321-9bda-b5cf338fd671.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure04a6b440fa244b20b3c6d11e17ab60fd = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure04a6b440-fa24-4b20-b3c6-d11e17ab60fd.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};