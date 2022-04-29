import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure8ce1553064344149b756696d8c41e9da = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure8ce15530-6434-4149-b756-696d8c41e9da.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureee12530e6dd8449a8db78391e157e524 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureee12530e-6dd8-449a-8db7-8391e157e524.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure0fd50077be754054b6bb05f1a2d405fe = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure0fd50077-be75-4054-b6bb-05f1a2d405fe.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};