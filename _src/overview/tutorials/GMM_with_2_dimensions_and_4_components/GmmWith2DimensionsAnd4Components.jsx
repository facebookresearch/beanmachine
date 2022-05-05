import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure5ac1598530994cdd86bb0bbd7443ce29 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5ac15985-3099-4cdd-86bb-0bbd7443ce29.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure4cbbd513b9424879a084160821f478ee = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4cbbd513-b942-4879-a084-160821f478ee.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure656f0ff306b346fba80011410c2a25a6 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure656f0ff3-06b3-46fb-a800-11410c2a25a6.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};