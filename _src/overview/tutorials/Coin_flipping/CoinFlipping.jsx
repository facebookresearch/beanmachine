import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure98a4a81e20a042acba2358afaae58fb9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure98a4a81e-20a0-42ac-ba23-58afaae58fb9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure83e3271f01bd48c2968199d1d192d52f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure83e3271f-01bd-48c2-9681-99d1d192d52f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};