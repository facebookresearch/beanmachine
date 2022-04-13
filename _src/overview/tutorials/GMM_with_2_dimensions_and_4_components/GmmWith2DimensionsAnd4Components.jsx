import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure9f4a2f34457b479b9e819381a84a2739 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9f4a2f34-457b-479b-9e81-9381a84a2739.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure5302393ba1874287beb4a47c955fbcd2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5302393b-a187-4287-beb4-a47c955fbcd2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureadc33e40e67c4f3db326abc0f664e2d5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureadc33e40-e67c-4f3d-b326-abc0f664e2d5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};