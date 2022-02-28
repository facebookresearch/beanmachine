import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure9c41139ddf69411985d1d66ced8f072a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9c41139d-df69-4119-85d1-d66ced8f072a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure565f28ac6359410abb3d4b661cc0fc65 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure565f28ac-6359-410a-bb3d-4b661cc0fc65.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};