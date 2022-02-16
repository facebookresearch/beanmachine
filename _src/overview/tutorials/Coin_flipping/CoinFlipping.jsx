import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure9851dc77ad174c8ca21a4445c7a68c57 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9851dc77-ad17-4c8c-a21a-4445c7a68c57.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure6365f3a791e04b4ea33b5ef06d1deabe = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6365f3a7-91e0-4b4e-a33b-5ef06d1deabe.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};