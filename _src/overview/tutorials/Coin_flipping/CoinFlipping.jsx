import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured25d1f50003f4fb08509a4372f806a98 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured25d1f50-003f-4fb0-8509-a4372f806a98.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec98846489f5a491a899208699e2ce246 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec9884648-9f5a-491a-8992-08699e2ce246.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};