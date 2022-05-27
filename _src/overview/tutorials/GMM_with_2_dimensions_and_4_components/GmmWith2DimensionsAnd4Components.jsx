import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured907e18bca844610869fb44442a01bce = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured907e18b-ca84-4610-869f-b44442a01bce.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec5c7a3edb2874a898d7e40894f351df0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec5c7a3ed-b287-4a89-8d7e-40894f351df0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure4d08c22bd35a4641a72bc01658716554 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4d08c22b-d35a-4641-a72b-c01658716554.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};