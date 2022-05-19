import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguref008879f63234a2283ed8eff3299352d = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref008879f-6323-4a22-83ed-8eff3299352d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigured306c8170fee4cdca6119d50bd304e15 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured306c817-0fee-4cdc-a611-9d50bd304e15.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure7c1e5206d6694b7fb1dd86800b171fc8 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7c1e5206-d669-4b7f-b1dd-86800b171fc8.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};