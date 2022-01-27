import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured2c7612b768145c69aec0d4ec0cdd9e0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured2c7612b-7681-45c6-9aec-0d4ec0cdd9e0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurebee1bec8095848a9942a97a0f7bd8041 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebee1bec8-0958-48a9-942a-97a0f7bd8041.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};