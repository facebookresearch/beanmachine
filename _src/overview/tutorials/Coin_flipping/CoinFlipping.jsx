import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure2b90f2061731479e9e6642ca9bf3ddf0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2b90f206-1731-479e-9e66-42ca9bf3ddf0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure80cb803767ca438a9c945384cf7a8da4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure80cb8037-67ca-438a-9c94-5384cf7a8da4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};