import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure4ab3d12b0ee4461f8dd609ce666b46cd = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4ab3d12b-0ee4-461f-8dd6-09ce666b46cd.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure3e7dbd853c9e422494f45969d874032d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3e7dbd85-3c9e-4224-94f4-5969d874032d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};