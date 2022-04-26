import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureb3a68c2476fd418aaf29b27312a5777b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb3a68c24-76fd-418a-af29-b27312a5777b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguredf3d728e0112477cbd74035efc1cdb17 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredf3d728e-0112-477c-bd74-035efc1cdb17.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};