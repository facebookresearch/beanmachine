import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure80a7a2c2cdcb4d81981e1f08025e982b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure80a7a2c2-cdcb-4d81-981e-1f08025e982b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureba0123dee75b4e2788c8a38274edefc3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureba0123de-e75b-4e27-88c8-a38274edefc3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};