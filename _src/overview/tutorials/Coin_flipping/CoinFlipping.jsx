import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure935fd8f470d64f31b51c072fbd184a57 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure935fd8f4-70d6-4f31-b51c-072fbd184a57.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure3d2742c0dfc1411786ee9075b6a56265 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3d2742c0-dfc1-4117-86ee-9075b6a56265.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};