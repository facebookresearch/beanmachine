import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured30c166c8ced4daebef3fd57efce123a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured30c166c-8ced-4dae-bef3-fd57efce123a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure88319eba61e84d239e36d87f4e82e2b4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure88319eba-61e8-4d23-9e36-d87f4e82e2b4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};