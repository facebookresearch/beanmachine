import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguref6be8eda9d114e798656b2edb77e9cbb = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref6be8eda-9d11-4e79-8656-b2edb77e9cbb.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigured39ba15b6c5243c488a5f5f00ba1fb06 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured39ba15b-6c52-43c4-88a5-f5f00ba1fb06.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};