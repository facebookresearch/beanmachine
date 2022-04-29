import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureb6bea0a003d04e44b282d898775acd0d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb6bea0a0-03d0-4e44-b282-d898775acd0d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure3086e31e02d749349c65117dd99383ed = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3086e31e-02d7-4934-9c65-117dd99383ed.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};