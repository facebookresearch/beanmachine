import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguref356e08f35314e348a3e0f705e9a63e3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref356e08f-3531-4e34-8a3e-0f705e9a63e3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure2b373b0a2bfd4559901a130827c996b6 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2b373b0a-2bfd-4559-901a-130827c996b6.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure45bd042c3b534c2aac4bbba66180229a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure45bd042c-3b53-4c2a-ac4b-bba66180229a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};