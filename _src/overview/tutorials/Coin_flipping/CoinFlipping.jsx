import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea13cbebc97ca41ac9632aaa23090699d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea13cbebc-97ca-41ac-9632-aaa23090699d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec4a790ce32ee4447bf5faaaaba80d19a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec4a790ce-32ee-4447-bf5f-aaaaba80d19a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};