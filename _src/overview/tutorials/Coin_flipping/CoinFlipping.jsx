import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure68b3a4b3229343a7911dec2853af1c79 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure68b3a4b3-2293-43a7-911d-ec2853af1c79.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure20fd5c095b824f1ba1ba278a20c70a2c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure20fd5c09-5b82-4f1b-a1ba-278a20c70a2c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};