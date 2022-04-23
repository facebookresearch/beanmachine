import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure6f01d26ebaec485cb17185b31ad9215d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure6f01d26e-baec-485c-b171-85b31ad9215d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure807e54f54da142fe8c378ec4aaf0b04f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure807e54f5-4da1-42fe-8c37-8ec4aaf0b04f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};