import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguredfd51cff1ec64e57a3f83f64328a9cbd = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredfd51cff-1ec6-4e57-a3f8-3f64328a9cbd.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec32803972a8f4f3a87eec51c9b1df9b2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec3280397-2a8f-4f3a-87ee-c51c9b1df9b2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure08ce05918a094632a211ea416122e6b9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure08ce0591-8a09-4632-a211-ea416122e6b9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};