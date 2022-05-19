import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea68643a773a2420480c24e18433f40e0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea68643a7-73a2-4204-80c2-4e18433f40e0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguredf7d629b9a5a4ccb9a2d3e38aaee9064 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredf7d629b-9a5a-4ccb-9a2d-3e38aaee9064.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure148b2b5b2e09487f883d6e21ec4a8016 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure148b2b5b-2e09-487f-883d-6e21ec4a8016.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};