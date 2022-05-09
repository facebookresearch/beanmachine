import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure7b850357a37f41fb8ad7be7c6d329f95 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7b850357-a37f-41fb-8ad7-be7c6d329f95.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguref127a4dccf704390afffe4817e42c91a = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref127a4dc-cf70-4390-afff-e4817e42c91a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure4b121289bbec447fad1007c2af39d490 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4b121289-bbec-447f-ad10-07c2af39d490.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};