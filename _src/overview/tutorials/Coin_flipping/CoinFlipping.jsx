import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure2711511a84fe44e581cd1dc3c84d60de = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2711511a-84fe-44e5-81cd-1dc3c84d60de.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec9c4f43c15414cd4a1f070522bcabaee = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec9c4f43c-1541-4cd4-a1f0-70522bcabaee.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};