import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure59720309df454e1db57fe4533e7c414a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure59720309-df45-4e1d-b57f-e4533e7c414a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure42f381f59967428ba551ab694df02439 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure42f381f5-9967-428b-a551-ab694df02439.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};