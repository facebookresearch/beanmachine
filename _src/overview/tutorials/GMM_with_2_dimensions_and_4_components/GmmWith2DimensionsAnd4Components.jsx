import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure40ddac23d0f549cfb2704316101c9358 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure40ddac23-d0f5-49cf-b270-4316101c9358.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure10ff900d68584c14b8211f736ce8a74f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure10ff900d-6858-4c14-b821-1f736ce8a74f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure5b50fbdac154416fbe323772a1731a6c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5b50fbda-c154-416f-be32-3772a1731a6c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};