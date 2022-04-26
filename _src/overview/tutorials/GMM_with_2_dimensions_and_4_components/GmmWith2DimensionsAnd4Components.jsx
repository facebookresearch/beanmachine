import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureb20efefa70b44a7f85538e50e3b6a453 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb20efefa-70b4-4a7f-8553-8e50e3b6a453.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure4f4609584f404af3a81f32c5682b1ccf = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4f460958-4f40-4af3-a81f-32c5682b1ccf.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguree5babef157ea409eab37fd6edb94bc21 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguree5babef1-57ea-409e-ab37-fd6edb94bc21.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};