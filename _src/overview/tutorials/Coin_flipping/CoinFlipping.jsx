import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure400619a8020f4b1fbd1487420d811595 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure400619a8-020f-4b1f-bd14-87420d811595.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigurec316f90a04204e3890274ac400ed742a = () => {
  const pathToData = "./assets/plot_data/PlotlyFigurec316f90a-0420-4e38-9027-4ac400ed742a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};