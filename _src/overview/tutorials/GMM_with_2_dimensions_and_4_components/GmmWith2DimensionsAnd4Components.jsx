import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure80ebdb5dc5084d0085f5b6f4a453e570 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure80ebdb5d-c508-4d00-85f5-b6f4a453e570.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure8a03b0b5c8be4284afb36cec8f17dda4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8a03b0b5-c8be-4284-afb3-6cec8f17dda4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure6e71f6d5d6914f04a86f4d34742539b7 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6e71f6d5-d691-4f04-a86f-4d34742539b7.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};