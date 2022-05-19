import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure3084b9420f134e18a5b69573468e8464 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3084b942-0f13-4e18-a5b6-9573468e8464.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure5b3d18ca43fd4277976bcf34608ee494 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5b3d18ca-43fd-4277-976b-cf34608ee494.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure08a856a5f0c34266b3df7988ec4bd81b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure08a856a5-f0c3-4266-b3df-7988ec4bd81b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};