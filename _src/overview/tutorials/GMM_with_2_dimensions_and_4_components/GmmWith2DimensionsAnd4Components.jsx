import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigureeb46a14ca685495aa4bfae1fd7281c00 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureeb46a14c-a685-495a-a4bf-ae1fd7281c00.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure414ab18cb81b41658b85ef756b5dca2c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure414ab18c-b81b-4165-8b85-ef756b5dca2c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurebf6be3e172064e30baed33ff67fd98e7 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebf6be3e1-7206-4e30-baed-33ff67fd98e7.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};