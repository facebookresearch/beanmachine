import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure6a257957ee5245a4a5ebc9e2ef27d887 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6a257957-ee52-45a4-a5eb-c9e2ef27d887.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure45f4275eff6e45db8347f7689272fff1 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure45f4275e-ff6e-45db-8347-f7689272fff1.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec2e89e800c1c437a9fdc5449438f602b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec2e89e80-0c1c-437a-9fdc-5449438f602b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};