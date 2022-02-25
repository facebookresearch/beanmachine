import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure7a0c5a56b38944efb84ef9818c8ff368 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7a0c5a56-b389-44ef-b84e-f9818c8ff368.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure41bb7259f18a4689814e772272875372 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure41bb7259-f18a-4689-814e-772272875372.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};