import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure712430fd8c904daeacf997765117822d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure712430fd-8c90-4dae-acf9-97765117822d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureca9f754feb174eab822649327e69a9ae = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureca9f754f-eb17-4eab-8226-49327e69a9ae.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};