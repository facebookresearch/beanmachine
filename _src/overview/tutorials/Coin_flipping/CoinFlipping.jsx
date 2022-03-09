import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure7c29dde8768f4d6ba29770861a65bde9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7c29dde8-768f-4d6b-a297-70861a65bde9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigured64bfa7e0b20447586de6fbfb16ed6f1 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured64bfa7e-0b20-4475-86de-6fbfb16ed6f1.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};