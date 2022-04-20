import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure0d086716a65e4640b7e245fb2e0e3d92 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure0d086716-a65e-4640-b7e2-45fb2e0e3d92.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure3398755132e24d94a13eacc27ee47035 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure33987551-32e2-4d94-a13e-acc27ee47035.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};