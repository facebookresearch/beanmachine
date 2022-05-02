import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured7315543b6e24894a58caf21d0ba7506 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured7315543-b6e2-4894-a58c-af21d0ba7506.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureaa38e8e7c8cf4e859554eeb0eb8e873e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureaa38e8e7-c8cf-4e85-9554-eeb0eb8e873e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};