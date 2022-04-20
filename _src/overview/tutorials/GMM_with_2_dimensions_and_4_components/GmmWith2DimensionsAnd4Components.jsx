import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurec077e9bc24e1415eb21fe2cbbb1c3399 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec077e9bc-24e1-415e-b21f-e2cbbb1c3399.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureff4bfea976f94da9abad4f2b18378ba0 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureff4bfea9-76f9-4da9-abad-4f2b18378ba0.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguref2636503f33b4403ae3a602591f7dd75 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref2636503-f33b-4403-ae3a-602591f7dd75.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};