import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureac0dfeb4e7254d4b9774c945f00386a6 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureac0dfeb4-e725-4d4b-9774-c945f00386a6.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure3a52e4a70a6e45e4bf4437b17ae0999a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3a52e4a7-0a6e-45e4-bf44-37b17ae0999a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};