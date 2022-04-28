import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure6fb0ae8532c444fda073dcb4e72833c3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6fb0ae85-32c4-44fd-a073-dcb4e72833c3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguree0947acd371e43b383fd0848953dc464 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguree0947acd-371e-43b3-83fd-0848953dc464.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};