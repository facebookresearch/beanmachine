import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure227e30e1439046b79ca785f49eeb1d38 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure227e30e1-4390-46b7-9ca7-85f49eeb1d38.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurea45f0876d16d432a82b75a9757289388 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea45f0876-d16d-432a-82b7-5a9757289388.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};