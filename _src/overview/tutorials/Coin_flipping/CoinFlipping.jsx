import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure40cc601fc2fa4f34a92e2d227b8f0e25 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure40cc601f-c2fa-4f34-a92e-2d227b8f0e25.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure1190472fd2d844bfa85d9fbdb64fab9b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure1190472f-d2d8-44bf-a85d-9fbdb64fab9b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};