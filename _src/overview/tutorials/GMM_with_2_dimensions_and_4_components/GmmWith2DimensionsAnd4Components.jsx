import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea51f18fa2eec419cad273f14d20fce29 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea51f18fa-2eec-419c-ad27-3f14d20fce29.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure0d1a54a735e449fba321d381dae9594a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure0d1a54a7-35e4-49fb-a321-d381dae9594a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureaf983455f03b421d8e38a9ee22ebfb03 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureaf983455-f03b-421d-8e38-a9ee22ebfb03.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};