import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurecafe4b10bbaf4fa1b62a78b75766c2c0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurecafe4b10-bbaf-4fa1-b62a-78b75766c2c0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure595defaff77f487eb74109cfe968bff3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure595defaf-f77f-487e-b741-09cfe968bff3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguree19fe1e0d29d4589a59e6b2fc54886eb = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguree19fe1e0-d29d-4589-a59e-6b2fc54886eb.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};