import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure88e5f9edf50f4f7981edbd7a1f910297 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure88e5f9ed-f50f-4f79-81ed-bd7a1f910297.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure676ea86103434e9892903c31eecf838d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure676ea861-0343-4e98-9290-3c31eecf838d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};