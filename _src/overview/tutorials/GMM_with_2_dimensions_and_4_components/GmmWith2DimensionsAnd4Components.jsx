import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea673c49663804e0fb2e8f11d36ee11a1 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea673c496-6380-4e0f-b2e8-f11d36ee11a1.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure13f87467f5cb4c67935d17f1549f3e11 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure13f87467-f5cb-4c67-935d-17f1549f3e11.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec7b8c2d1156148f9a571ca3ea7d17fb2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec7b8c2d1-1561-48f9-a571-ca3ea7d17fb2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};