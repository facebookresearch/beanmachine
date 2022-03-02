import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure830d8dca7a764268b5b83cfae629d27b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure830d8dca-7a76-4268-b5b8-3cfae629d27b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure6f1e059e05e745c19556a5f2651db96a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6f1e059e-05e7-45c1-9556-a5f2651db96a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};