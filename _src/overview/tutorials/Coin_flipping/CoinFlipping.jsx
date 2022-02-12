import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurebd5914ed9b344a08b3a3d9df77b611d5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebd5914ed-9b34-4a08-b3a3-d9df77b611d5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure39ea8aa12e284b22937cf529feae0ad4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure39ea8aa1-2e28-4b22-937c-f529feae0ad4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};