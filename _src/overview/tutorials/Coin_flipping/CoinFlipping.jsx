import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureb291f607dcbd4a5290f877dffd0371d0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb291f607-dcbd-4a52-90f8-77dffd0371d0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure317a17b16b5c4e13a2b0f57228be33a5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure317a17b1-6b5c-4e13-a2b0-f57228be33a5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};