import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguredb5ea36fd28c423484b73fb4ddbfc54c = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredb5ea36f-d28c-4234-84b7-3fb4ddbfc54c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure955b46bff3c240c8a39449a465adcc3c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure955b46bf-f3c2-40c8-a394-49a465adcc3c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};