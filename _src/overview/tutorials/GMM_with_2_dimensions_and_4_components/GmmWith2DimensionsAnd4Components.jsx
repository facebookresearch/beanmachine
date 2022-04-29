import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFiguref9146ec676354895b059e81f49307786 = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguref9146ec6-7635-4895-b059-e81f49307786.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFiguree3ebfa4636f14d039b70472c19b53d5a = () => {
  const pathToData = "./assets/plot_data/PlotlyFiguree3ebfa46-36f1-4d03-9b70-472c19b53d5a.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure762f83b5261b4ac8afbb95160754e35c = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure762f83b5-261b-4ac8-afbb-95160754e35c.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};