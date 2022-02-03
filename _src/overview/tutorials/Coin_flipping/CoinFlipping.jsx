import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure2d04a14a51ab4f0d83766da7f4cc0b2e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2d04a14a-51ab-4f0d-8376-6da7f4cc0b2e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurecc3c54bc11c0464ba44054c7c96a9db4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurecc3c54bc-11c0-464b-a440-54c7c96a9db4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};