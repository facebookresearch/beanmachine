import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea538a138eae4489ba6863e4732ec3638 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea538a138-eae4-489b-a686-3e4732ec3638.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure8113db77f58b43248c912dd8ddf5d859 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8113db77-f58b-4324-8c91-2dd8ddf5d859.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurea27c47661f13411babd89939f1d92dd9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea27c4766-1f13-411b-abd8-9939f1d92dd9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};