import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurebf70911b87744b09b12affc3f5026f13 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurebf70911b-8774-4b09-b12a-ffc3f5026f13.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure2a2bcd72fc274e3db69ba8ac292a683b = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2a2bcd72-fc27-4e3d-b69b-a8ac292a683b.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};