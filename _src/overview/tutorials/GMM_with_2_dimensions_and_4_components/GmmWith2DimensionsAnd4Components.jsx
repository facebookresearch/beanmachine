import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurebe500e2fe8994bb8abdd4975e2b10207 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebe500e2f-e899-4bb8-abdd-4975e2b10207.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec24005e912a64d14b3ee9d48ba4d29d5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec24005e9-12a6-4d14-b3ee-9d48ba4d29d5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureab185f67b12049278fe1e943e3207d50 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureab185f67-b120-4927-8fe1-e943e3207d50.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};