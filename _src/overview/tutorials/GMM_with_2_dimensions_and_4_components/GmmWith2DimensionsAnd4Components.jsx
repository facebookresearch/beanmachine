import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured9589f9568674edc813be2606e4d6739 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured9589f95-6867-4edc-813b-e2606e4d6739.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure4d99f5fcd9e54862a8ec3f0558c7fdad = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4d99f5fc-d9e5-4862-a8ec-3f0558c7fdad.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure24dc69917ee749619d2057406a7f7387 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure24dc6991-7ee7-4961-9d20-57406a7f7387.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};