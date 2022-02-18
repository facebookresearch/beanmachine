import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure4f9473aeb5f14376a01db1241180863e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4f9473ae-b5f1-4376-a01d-b1241180863e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure3c8fd6789b77438a8a4a9c842cba6970 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3c8fd678-9b77-438a-8a4a-9c842cba6970.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};