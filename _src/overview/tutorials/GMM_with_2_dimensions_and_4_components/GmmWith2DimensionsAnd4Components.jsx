import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea4d9c2c033614560a362bc0b8ab43915 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea4d9c2c0-3361-4560-a362-bc0b8ab43915.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure1286fd237b7a483ab09ec9471689857f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure1286fd23-7b7a-483a-b09e-c9471689857f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguredfa4f2f201bf4cf98814a14fc8dfc501 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguredfa4f2f2-01bf-4cf9-8814-a14fc8dfc501.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};