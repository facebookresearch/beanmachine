import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure93a00bd0368443ee9fab0a9d797294e2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure93a00bd0-3684-43ee-9fab-0a9d797294e2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure890758f3ac394e1a8233ac9fc9a788a3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure890758f3-ac39-4e1a-8233-ac9fc9a788a3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};