import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured0045e402cd4493e8fcd34e9a3839172 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured0045e40-2cd4-493e-8fcd-34e9a3839172.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguredea19395f1604e76be42662be4bba5b7 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredea19395-f160-4e76-be42-662be4bba5b7.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};