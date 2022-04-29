import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure5c7b79046b51416c8a749a3027d5b00b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5c7b7904-6b51-416c-8a74-9a3027d5b00b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure246b0ca46b304210a72fdbb6849564ad = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure246b0ca4-6b30-4210-a72f-dbb6849564ad.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};