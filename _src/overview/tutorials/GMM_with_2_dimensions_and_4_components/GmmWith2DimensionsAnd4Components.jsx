import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure1ecd3f9562924e449c9c4b906fdb681b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure1ecd3f95-6292-4e44-9c9c-4b906fdb681b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure2d0f7e6565e34bd284ca248ef2ef61c1 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2d0f7e65-65e3-4bd2-84ca-248ef2ef61c1.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureab1e2b628bd74e44b99a71f7dfca13c7 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureab1e2b62-8bd7-4e44-b99a-71f7dfca13c7.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};