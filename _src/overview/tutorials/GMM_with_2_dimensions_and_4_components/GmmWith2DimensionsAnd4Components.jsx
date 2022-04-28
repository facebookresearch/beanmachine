import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure48e3b95dbcf643559ea42de64048f3dd = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure48e3b95d-bcf6-4355-9ea4-2de64048f3dd.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurea7b8b9b8b4a2459f88a200e3195dc4a1 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea7b8b9b8-b4a2-459f-88a2-00e3195dc4a1.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguree7a1b7fc2f594b5bba340c4ce8a7c902 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguree7a1b7fc-2f59-4b5b-ba34-0c4ce8a7c902.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};