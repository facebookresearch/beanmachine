import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure4a787f4bc72b49848bcb1404c87888c6 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4a787f4b-c72b-4984-8bcb-1404c87888c6.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure330e106a2a5f4ca08d86a91b7e3dfdb9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure330e106a-2a5f-4ca0-8d86-a91b7e3dfdb9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};