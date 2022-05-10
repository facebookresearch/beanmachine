import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure201be2d52e834ea984c90d2dae4098bf = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure201be2d5-2e83-4ea9-84c9-0d2dae4098bf.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureeffac44cecc54cdc8ec2e73eb816bfdc = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureeffac44c-ecc5-4cdc-8ec2-e73eb816bfdc.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure8cc42f63c87c446683c8a099505f5ad9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8cc42f63-c87c-4466-83c8-a099505f5ad9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};