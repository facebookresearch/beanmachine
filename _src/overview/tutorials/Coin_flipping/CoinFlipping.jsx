import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureefe1efc4bcde494eb298b85cccb373c0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureefe1efc4-bcde-494e-b298-b85cccb373c0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure3fb8f53557384a3ea432d4675c12e124 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3fb8f535-5738-4a3e-a432-d4675c12e124.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};