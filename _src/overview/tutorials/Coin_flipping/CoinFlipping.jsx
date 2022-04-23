import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured3e10700f4de4d938ce3771383d33ee2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured3e10700-f4de-4d93-8ce3-771383d33ee2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure9e828b5670a34ad0bd215296e6d0e3b2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9e828b56-70a3-4ad0-bd21-5296e6d0e3b2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};