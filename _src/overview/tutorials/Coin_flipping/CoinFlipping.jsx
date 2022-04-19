import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguref38f81cbf8fd498b859ea5204a0695bb = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref38f81cb-f8fd-498b-859e-a5204a0695bb.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure4a4175b6d42e4135a2848d2f0099e8c2 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4a4175b6-d42e-4135-a284-8d2f0099e8c2.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};