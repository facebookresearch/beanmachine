import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguree09a367f08a24fa8a79b55320d8ad87f = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguree09a367f-08a2-4fa8-a79b-55320d8ad87f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure5488856226ea444e944ebec066a3abe5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure54888562-26ea-444e-944e-bec066a3abe5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};