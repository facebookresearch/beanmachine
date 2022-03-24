import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure42f6d349e29e473296f6118ec00354f3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure42f6d349-e29e-4732-96f6-118ec00354f3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure202f2fd80d7d4d85ac590cac8e7a1ec5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure202f2fd8-0d7d-4d85-ac59-0cac8e7a1ec5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};