import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure1adca063293d445bb9dfc68472c05386 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure1adca063-293d-445b-b9df-c68472c05386.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure2a1b918404df41308291432077925dab = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2a1b9184-04df-4130-8291-432077925dab.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};