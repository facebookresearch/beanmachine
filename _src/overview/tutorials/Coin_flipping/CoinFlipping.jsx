import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure595a123460834e86987788464d8a85ee = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure595a1234-6083-4e86-9877-88464d8a85ee.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure16569c901bbb4c7bb28ce3da2731530f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure16569c90-1bbb-4c7b-b28c-e3da2731530f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};