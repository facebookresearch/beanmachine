import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure06a274c405ce4b37895047697c5494fe = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure06a274c4-05ce-4b37-8950-47697c5494fe.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure5aae2ec57860470e904e4b052ab91b68 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5aae2ec5-7860-470e-904e-4b052ab91b68.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure45c3fbd17ddc40cfb4c0068ed3f7421e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure45c3fbd1-7ddc-40cf-b4c0-068ed3f7421e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};