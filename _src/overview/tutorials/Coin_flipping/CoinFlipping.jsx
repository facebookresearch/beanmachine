import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure9fe116bc9a7640328a2dd78c00b5ab77 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9fe116bc-9a76-4032-8a2d-d78c00b5ab77.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureaf65b313ac49481b8759f5cdd4a7c9f1 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureaf65b313-ac49-481b-8759-f5cdd4a7c9f1.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};