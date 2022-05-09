import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure9bb2971aa2b14679a52fc13e4bbb0555 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9bb2971a-a2b1-4679-a52f-c13e4bbb0555.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure2e0d79ca3c6d496a83654bfc42aa2422 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2e0d79ca-3c6d-496a-8365-4bfc42aa2422.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};