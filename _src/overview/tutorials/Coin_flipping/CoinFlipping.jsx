import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure5d2468a860514198a221622e1c887306 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5d2468a8-6051-4198-a221-622e1c887306.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec482f55c2310464bb2ee229fe034f161 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec482f55c-2310-464b-b2ee-229fe034f161.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};