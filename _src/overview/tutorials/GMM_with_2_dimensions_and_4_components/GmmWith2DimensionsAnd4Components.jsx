import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea0a74e81903b4b0b950fcc0e778165b9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea0a74e81-903b-4b0b-950f-cc0e778165b9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure2e8370347d65489389329f80ec7087f5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2e837034-7d65-4893-8932-9f80ec7087f5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure8a2b79144a024054922d6af6cdbef381 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8a2b7914-4a02-4054-922d-6af6cdbef381.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};