import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurecc8be3a732e94afd86e0ea9cbb3d1264 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurecc8be3a7-32e9-4afd-86e0-ea9cbb3d1264.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguref1b9f08e6e7445dfb16fbd779d611299 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref1b9f08e-6e74-45df-b16f-bd779d611299.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};