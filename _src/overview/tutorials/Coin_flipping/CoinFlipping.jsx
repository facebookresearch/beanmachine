import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure4abb97e1942948c295bcc1c493d4e36e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure4abb97e1-9429-48c2-95bc-c1c493d4e36e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure7236531e5b7e470c88488ba6211e415f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure7236531e-5b7e-470c-8848-8ba6211e415f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};