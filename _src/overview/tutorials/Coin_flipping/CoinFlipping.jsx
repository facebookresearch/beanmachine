import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure6f95b253b12b4ed68a6c1943a4cee38e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6f95b253-b12b-4ed6-8a6c-1943a4cee38e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure8ee86c3839b54b939c0fd87c4e81c7ef = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8ee86c38-39b5-4b93-9c0f-d87c4e81c7ef.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};