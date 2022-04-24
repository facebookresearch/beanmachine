import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurede2ce1ce85f540bdb388fda7ad3e7beb = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurede2ce1ce-85f5-40bd-b388-fda7ad3e7beb.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurebd9f24f0d33d485cb1583e34211a2077 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebd9f24f0-d33d-485c-b158-3e34211a2077.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};