import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure1a66b368c00c48f39bf53098be9e6bbd = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure1a66b368-c00c-48f3-9bf5-3098be9e6bbd.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurebf34819e04384d4a9b3b6fb9ada6ccb2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebf34819e-0438-4d4a-9b3b-6fb9ada6ccb2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec633322c7a7d4b84a9b166f4ebbdf712 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec633322c-7a7d-4b84-a9b1-66f4ebbdf712.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};