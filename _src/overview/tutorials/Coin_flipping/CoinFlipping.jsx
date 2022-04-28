import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigure3856eb984be548e3929e6de1e306ff3f = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure3856eb98-4be5-48e3-929e-6de1e306ff3f.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure04ec29489d6f41c4a175fcdc0a21d078 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure04ec2948-9d6f-41c4-a175-fcdc0a21d078.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};