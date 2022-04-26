import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureafebfda81fca40e187390cc7bebdecbb = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureafebfda8-1fca-40e1-8739-0cc7bebdecbb.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureafe36485ce724a7ab2ff0409e2f41946 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureafe36485-ce72-4a7a-b2ff-0409e2f41946.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure5dff58d36e9b4a008a901d0d935d0377 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5dff58d3-6e9b-4a00-8a90-1d0d935d0377.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};