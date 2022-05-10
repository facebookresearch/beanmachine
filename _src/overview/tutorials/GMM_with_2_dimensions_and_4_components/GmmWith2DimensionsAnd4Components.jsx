import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureea02b289807c41f5b58be15024061441 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureea02b289-807c-41f5-b58b-e15024061441.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure011f45e521ea492f99836f174f48eb13 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure011f45e5-21ea-492f-9983-6f174f48eb13.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure0befd4fb33c3496d8c7e3213fae0d61b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure0befd4fb-33c3-496d-8c7e-3213fae0d61b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};