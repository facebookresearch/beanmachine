import React from "react";
import { BokehFigure } from "../../../website/src/components/Plotting.jsx";

export const PriorPoissonPlot = () => {
  const pathToData = "./prior_poisson.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />;
};

export const PriorExponentialPlot = () => {
  const pathToData = "./prior_exponential.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />;
};

export const PosteriorRateStaticPlot = () => {
  const pathToData = "./posterior_rate_static.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />;
};
