import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure14edbf93eaee4f48a63d1c5de294936e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure14edbf93-eaee-4f48-a63d-1c5de294936e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure07a69f470770451eb3b1308fab19c274 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure07a69f47-0770-451e-b3b1-308fab19c274.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurea1f95fff2efd4b448eb442a091eaec67 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea1f95fff-2efd-4b44-8eb4-42a091eaec67.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};