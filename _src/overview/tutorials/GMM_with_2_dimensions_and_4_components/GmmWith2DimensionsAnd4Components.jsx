import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureb4ec6627f10e4383a0da3b27d547f770 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb4ec6627-f10e-4383-a0da-3b27d547f770.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure04e25f5c01b346c79ca26c663c9dbf50 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure04e25f5c-01b3-46c7-9ca2-6c663c9dbf50.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure6804fd572e514d19b8242b1038a0f80a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6804fd57-2e51-4d19-b824-2b1038a0f80a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};