import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure2d97004ca7124f0dae105d13504614db = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure2d97004c-a712-4f0d-ae10-5d13504614db.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure9ef03b0c4e20428490beaadc865c167d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9ef03b0c-4e20-4284-90be-aadc865c167d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure9cb090c5f4c74753a834a2235f4efb1a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure9cb090c5-f4c7-4753-a834-a2235f4efb1a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};