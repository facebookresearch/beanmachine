import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure38f28d2e0f904f8db194bc5464218e01 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure38f28d2e-0f90-4f8d-b194-bc5464218e01.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure896c4fc3f8f84ef98e09d6beb6d820ad = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure896c4fc3-f8f8-4ef9-8e09-d6beb6d820ad.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};