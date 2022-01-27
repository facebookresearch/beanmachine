import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure673c3f768e3448d9ad82fca7e19d2baa = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure673c3f76-8e34-48d9-ad82-fca7e19d2baa.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguredb71bb3d6eac4b85b24413aaf26f510a = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredb71bb3d-6eac-4b85-b244-13aaf26f510a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};