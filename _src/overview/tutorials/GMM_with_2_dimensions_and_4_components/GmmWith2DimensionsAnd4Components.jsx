import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure8a780a6945df4066a8cd35e23b0901fe = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8a780a69-45df-4066-a8cd-35e23b0901fe.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurea6a8ccf2c0454c11bd2c2ad6a9c7180b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea6a8ccf2-c045-4c11-bd2c-2ad6a9c7180b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurea47f6f7fdd4b4cfaaa5140276ba1f872 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea47f6f7f-dd4b-4cfa-aa51-40276ba1f872.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};