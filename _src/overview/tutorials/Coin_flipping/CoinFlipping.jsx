import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure789ce4c54f664dc792c35b9771f4f19c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure789ce4c5-4f66-4dc7-92c3-5b9771f4f19c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure2cb722c0d778445aae0527bdd0486d8f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2cb722c0-d778-445a-ae05-27bdd0486d8f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};