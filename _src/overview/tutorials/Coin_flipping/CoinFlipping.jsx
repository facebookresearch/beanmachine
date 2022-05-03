import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurebd8a07c0fa0a4443a1db74fe3e90727d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebd8a07c0-fa0a-4443-a1db-74fe3e90727d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure8846704e8d8d461d9bb5ac7f9c56eb7a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8846704e-8d8d-461d-9bb5-ac7f9c56eb7a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};