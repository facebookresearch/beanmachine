import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure3df0e0e8d2044fca808d79e222e71b6e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3df0e0e8-d204-4fca-808d-79e222e71b6e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguref49d78bbcbf9460891085b6d7ae73178 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref49d78bb-cbf9-4608-9108-5b6d7ae73178.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};