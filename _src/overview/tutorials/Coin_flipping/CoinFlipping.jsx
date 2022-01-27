import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurefc190ad786194da391419c94b114a1c2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurefc190ad7-8619-4da3-9141-9c94b114a1c2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure9ff70f8def40444193b45570fffa95a4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9ff70f8d-ef40-4441-93b4-5570fffa95a4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};