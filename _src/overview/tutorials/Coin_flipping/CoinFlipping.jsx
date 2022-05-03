import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure078e7f2200eb4143b83593369e2f2416 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure078e7f22-00eb-4143-b835-93369e2f2416.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure104b2e2067f64dc3a95fcbf83904cad5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure104b2e20-67f6-4dc3-a95f-cbf83904cad5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};