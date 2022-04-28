import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure7a718d789070403f8ead20d84668890c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7a718d78-9070-403f-8ead-20d84668890c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure7307f1c8a4ba4391a0c5020ef7c39ade = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7307f1c8-a4ba-4391-a0c5-020ef7c39ade.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};