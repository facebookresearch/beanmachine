import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured9b401096d3f4f6e8703a2c16575d219 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured9b40109-6d3f-4f6e-8703-a2c16575d219.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurecfabe8cfd4334c799ffe5fd506c55baa = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurecfabe8cf-d433-4c79-9ffe-5fd506c55baa.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};