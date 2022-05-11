import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure946e614bcf114ddd9d7cef7789bfd86f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure946e614b-cf11-4ddd-9d7c-ef7789bfd86f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurebcd18f857e344fe09f70b50e81c7a69c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebcd18f85-7e34-4fe0-9f70-b50e81c7a69c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigured37f5059b2e544d4a42bc074a0f6e383 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured37f5059-b2e5-44d4-a42b-c074a0f6e383.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};