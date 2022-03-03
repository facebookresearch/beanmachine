import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurebf4a896cd9614795b8d73f3a477bec15 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebf4a896c-d961-4795-b8d7-3f3a477bec15.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure833e4ebb8bfa48d393f765916c267b9f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure833e4ebb-8bfa-48d3-93f7-65916c267b9f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};