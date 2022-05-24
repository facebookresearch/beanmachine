import React from 'react';
import { PlotlyFigure } from '../../../../website/src/components/Plotting.jsx';

export const PlotlyFigured62b9ce42a35447d872631bd6f9743f4 = () => {
  const pathToData = "./assets/plot_data/PlotlyFigured62b9ce4-2a35-447d-8726-31bd6f9743f4.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigureeee247efd77b472180013c235f659d8d = () => {
  const pathToData = "./assets/plot_data/PlotlyFigureeee247ef-d77b-4721-8001-3c235f659d8d.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};

export const PlotlyFigure35470af32d1f44dbbc45982766b1421e = () => {
  const pathToData = "./assets/plot_data/PlotlyFigure35470af3-2d1f-44db-bc45-982766b1421e.json";
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const data = plotData["data"];
  const layout = plotData["layout"];
  return <PlotlyFigure data={data} layout={layout} />
};