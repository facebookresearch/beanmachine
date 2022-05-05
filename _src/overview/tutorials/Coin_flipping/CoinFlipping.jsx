import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguref9fd4308e2784741b92a7862530febf4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref9fd4308-e278-4741-b92a-7862530febf4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec79933fbc5564c8890789733e69daa1f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec79933fb-c556-4c88-9078-9733e69daa1f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};