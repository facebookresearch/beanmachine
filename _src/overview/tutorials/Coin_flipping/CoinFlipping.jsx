import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure954d0d73713742cdaf88b605815b6cdf = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure954d0d73-7137-42cd-af88-b605815b6cdf.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure8f856fe7a3ed4a2b8d581289e6333ecc = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8f856fe7-a3ed-4a2b-8d58-1289e6333ecc.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};