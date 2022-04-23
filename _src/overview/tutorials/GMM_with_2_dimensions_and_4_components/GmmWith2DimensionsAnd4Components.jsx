import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurecba1e3e14b8847299af48af419172cf4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurecba1e3e1-4b88-4729-9af4-8af419172cf4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureda09d31bb9604d2e801dba4ec2ba1c4c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureda09d31b-b960-4d2e-801d-ba4ec2ba1c4c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure45821ad94a13482cbd59e504628f4f3d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure45821ad9-4a13-482c-bd59-e504628f4f3d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};