import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurefd92fcad30ec4c97a2ab9098f59be517 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurefd92fcad-30ec-4c97-a2ab-9098f59be517.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureb0b14c3c56d840f493d398057ca22f0d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb0b14c3c-56d8-40f4-93d3-98057ca22f0d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};