import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigurec3741ee8fcd643e894136119209355f5 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec3741ee8-fcd6-43e8-9413-6119209355f5.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure913c3ea51eb745e4a12523c6fb36e91f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure913c3ea5-1eb7-45e4-a125-23c6fb36e91f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure5bf6b1d1ab144ad9aadb67c7641fe43c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure5bf6b1d1-ab14-4ad9-aadb-67c7641fe43c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};