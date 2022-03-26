import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure29dc408a2bb8417083cb2eb3a411f24e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure29dc408a-2bb8-4170-83cb-2eb3a411f24e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure08d4f019f855488b958fb574d1282e20 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure08d4f019-f855-488b-958f-b574d1282e20.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};