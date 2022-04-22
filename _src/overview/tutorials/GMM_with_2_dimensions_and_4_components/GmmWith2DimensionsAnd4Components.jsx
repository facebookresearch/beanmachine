import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure3d7470ba7df44bcf81b8c3fd89d02518 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3d7470ba-7df4-4bcf-81b8-c3fd89d02518.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure7f95756e2898424c82ffea010b89621c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7f95756e-2898-424c-82ff-ea010b89621c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure62227068586a4efd8111777f2b3c9bee = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure62227068-586a-4efd-8111-777f2b3c9bee.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};