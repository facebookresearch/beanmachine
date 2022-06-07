import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguref594bda319b94699b01c67260e7f13e7 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref594bda3-19b9-4699-b01c-67260e7f13e7.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurefb980105b565487c86b91acb2ba03fe3 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurefb980105-b565-487c-86b9-1acb2ba03fe3.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure6e140dd8af7848748a5e4784ce7a86b7 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6e140dd8-af78-4874-8a5e-4784ce7a86b7.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};