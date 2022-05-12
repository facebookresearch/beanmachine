import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguree4324f7737bf47849269ecc0e6b8dc2a = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguree4324f77-37bf-4784-9269-ecc0e6b8dc2a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure851bc812fe4b49f58a921ab759cb47be = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure851bc812-fe4b-49f5-8a92-1ab759cb47be.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure23db5ecdb7e642ed9b462467b4045c82 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure23db5ecd-b7e6-42ed-9b46-2467b4045c82.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};