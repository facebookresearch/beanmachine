import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure78a80e181a274b36b446e030ff823613 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure78a80e18-1a27-4b36-b446-e030ff823613.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure017500506ed04bd086902d4be5491fff = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure01750050-6ed0-4bd0-8690-2d4be5491fff.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};