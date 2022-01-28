import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurece28eab5731e4e6cb27893751450bde5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurece28eab5-731e-4e6c-b278-93751450bde5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure837a6b02887844b681250956a3a74beb = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure837a6b02-8878-44b6-8125-0956a3a74beb.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};