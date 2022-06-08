import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguread02e02e35c84d3680de0f30f9600847 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguread02e02e-35c8-4d36-80de-0f30f9600847.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure4469645d499943febb110d106236e5e2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4469645d-4999-43fe-bb11-0d106236e5e2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure46e6d02ea8de46eca31f360546c2c0de = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure46e6d02e-a8de-46ec-a31f-360546c2c0de.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};