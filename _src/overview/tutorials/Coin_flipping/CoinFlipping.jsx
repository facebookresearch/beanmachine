import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure409ba6e2bb6f4dc0b1d78d98131d4035 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure409ba6e2-bb6f-4dc0-b1d7-8d98131d4035.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure88b4ee2a09184c2e92d620f1d2e17727 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure88b4ee2a-0918-4c2e-92d6-20f1d2e17727.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};