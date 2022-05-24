import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguref8042e01e40c47babaae8575dfc18221 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref8042e01-e40c-47ba-baae-8575dfc18221.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureb029927fba8b40cda853d0bd8947d393 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureb029927f-ba8b-40cd-a853-d0bd8947d393.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure62b0aa49c65d4324939d4053e702cced = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure62b0aa49-c65d-4324-939d-4053e702cced.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};