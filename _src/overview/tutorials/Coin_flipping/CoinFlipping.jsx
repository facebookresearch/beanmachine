import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure016e141293184f1fb6381d5de478855f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure016e1412-9318-4f1f-b638-1d5de478855f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure3488ab9a5dd540d5bae26a14821601b2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3488ab9a-5dd5-40d5-bae2-6a14821601b2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};