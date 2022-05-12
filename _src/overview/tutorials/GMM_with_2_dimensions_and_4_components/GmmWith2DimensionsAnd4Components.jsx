import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure07365e0862d1459990de99fbaa2a8be0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure07365e08-62d1-4599-90de-99fbaa2a8be0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurefc91989525af43f6aac4a9f4709d9537 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurefc919895-25af-43f6-aac4-a9f4709d9537.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec1062cb385334e79b45bd04eadb6c838 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec1062cb3-8533-4e79-b45b-d04eadb6c838.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};