import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure1cfa0fea1f1a4149a14be5c18c5a19dd = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure1cfa0fea-1f1a-4149-a14b-e5c18c5a19dd.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure69d1c0a98af648f4ab58c5e480098516 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure69d1c0a9-8af6-48f4-ab58-c5e480098516.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure00273a634a00473ba4fb1672c4ac8f59 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure00273a63-4a00-473b-a4fb-1672c4ac8f59.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};