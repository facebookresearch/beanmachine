import React from "react";
import { BokehFigure } from "../../../website/src/components/Plotting.jsx";

export const PriorPoissonIntroPlot = () => {
  const pathToData = "./prior_poisson_intro.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />;
};

export const PriorExponentialPlot = () => {
  const pathToData = "./prior_exponential.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />;
};
