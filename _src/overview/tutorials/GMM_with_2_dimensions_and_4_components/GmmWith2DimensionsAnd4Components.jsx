import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguredf7b07d530a745d386b8dc98324b483e = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredf7b07d5-30a7-45d3-86b8-dc98324b483e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec9e59d3f5fd64192bf8df15779e4ce61 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec9e59d3f-5fd6-4192-bf8d-f15779e4ce61.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure80cfa3b5428b4102a36fdb9445d107f3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure80cfa3b5-428b-4102-a36f-db9445d107f3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};