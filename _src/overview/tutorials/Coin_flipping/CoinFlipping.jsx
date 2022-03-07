import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure220ca6d525384e9eb5389d508f0a77c4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure220ca6d5-2538-4e9e-b538-9d508f0a77c4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure0968e10982c2478db5b89d6321808e04 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure0968e109-82c2-478d-b5b8-9d6321808e04.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};