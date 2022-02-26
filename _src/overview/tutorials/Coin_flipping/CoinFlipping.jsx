import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure601d5268fc2b4d44b22c0e75f7aba0a2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure601d5268-fc2b-4d44-b22c-0e75f7aba0a2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurebd8bef28a3134bec8210cd4fc971391c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebd8bef28-a313-4bec-8210-cd4fc971391c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};