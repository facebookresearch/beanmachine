import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure69b83af1f2fa49dc88804306bc2bd01a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure69b83af1-f2fa-49dc-8880-4306bc2bd01a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure0f1ffa302a9341e18c6ac26ccd0fc999 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure0f1ffa30-2a93-41e1-8c6a-c26ccd0fc999.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};