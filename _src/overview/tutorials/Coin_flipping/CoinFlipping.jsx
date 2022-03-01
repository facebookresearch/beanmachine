import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure5dbf9f7e1aee4318997392f4a5c45c61 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5dbf9f7e-1aee-4318-9973-92f4a5c45c61.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure7880b8d1754b44a090b781bebd6892cd = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7880b8d1-754b-44a0-90b7-81bebd6892cd.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};