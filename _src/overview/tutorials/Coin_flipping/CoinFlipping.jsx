import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure414080a3536047799870c34bc57f1f4d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure414080a3-5360-4779-9870-c34bc57f1f4d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec3923a2875fb40ab94dc1cef12bfd0cd = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec3923a28-75fb-40ab-94dc-1cef12bfd0cd.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};