import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure7dbd4accd71c4645b5db6df405209ad1 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7dbd4acc-d71c-4645-b5db-6df405209ad1.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguredbc09272ec8c403f91834e67cc1f8a13 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredbc09272-ec8c-403f-9183-4e67cc1f8a13.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure3ac831d6dc434433bf161b22983159f3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3ac831d6-dc43-4433-bf16-1b22983159f3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};