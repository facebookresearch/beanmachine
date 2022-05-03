import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurec2449dc978b049878e28d8e09eb732a1 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec2449dc9-78b0-4987-8e28-d8e09eb732a1.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure5f04e7120a4f4aa6bc44494f7508844c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5f04e712-0a4f-4aa6-bc44-494f7508844c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureed2216e83dc24798ab5c84a936e0670c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureed2216e8-3dc2-4798-ab5c-84a936e0670c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};