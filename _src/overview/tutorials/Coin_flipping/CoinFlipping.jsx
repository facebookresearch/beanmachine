import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured6d3f2100a8b4043bc600da7ecac864a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured6d3f210-0a8b-4043-bc60-0da7ecac864a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurecf8ebf887feb4d1e81bb6b7e4ecc7f0b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurecf8ebf88-7feb-4d1e-81bb-6b7e4ecc7f0b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};