import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea7993c9c366d4013b084a301ec5af6c9 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea7993c9c-366d-4013-b084-a301ec5af6c9.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguref396e4fb7ed44db2af9070161275cd48 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref396e4fb-7ed4-4db2-af90-70161275cd48.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigured0fa10882001453ab652f6c2c6673730 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured0fa1088-2001-453a-b652-f6c2c6673730.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};