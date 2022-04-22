import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurea7f32945fa554ced836b3bc21e23e65d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurea7f32945-fa55-4ced-836b-3bc21e23e65d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure6d3e876b346247eea5256baf69153f14 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6d3e876b-3462-47ee-a525-6baf69153f14.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};