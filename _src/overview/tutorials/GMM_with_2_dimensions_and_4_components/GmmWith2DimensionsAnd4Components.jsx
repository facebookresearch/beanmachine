import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguredd539581a9b74418a1c69d9992e85dcd = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredd539581-a9b7-4418-a1c6-9d9992e85dcd.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureac599564731c4321a4338f3c18e18a0b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureac599564-731c-4321-a433-8f3c18e18a0b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure0b2bd0549a2f49d1ad50f363eb2e757d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure0b2bd054-9a2f-49d1-ad50-f363eb2e757d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};