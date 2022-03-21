import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureadf5a1c7fc1149da9ebbd2ec217fe0ac = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureadf5a1c7-fc11-49da-9ebb-d2ec217fe0ac.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure9bc6cea279d94462ace01c3441e86917 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9bc6cea2-79d9-4462-ace0-1c3441e86917.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};