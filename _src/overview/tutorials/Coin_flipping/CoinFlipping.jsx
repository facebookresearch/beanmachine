import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureed6c3ebaab1e43fe9ee9c6018ce3ac0d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureed6c3eba-ab1e-43fe-9ee9-c6018ce3ac0d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure1b049b3a6b4248e8b6f9120320e54b4a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure1b049b3a-6b42-48e8-b6f9-120320e54b4a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};