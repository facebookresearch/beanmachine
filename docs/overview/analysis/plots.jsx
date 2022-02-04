import React from "react";
import { BokehFigure } from "../../../website/src/components/Plotting.jsx";

export const PosteriorRateDynamicPlot = () => {
  const pathToData = "./posterior_rate_dynamic.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />;
};

export const MCMCTracePlot = () => {
  const pathToData = "./mcmc_trace.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />;
};

export const MCMCAutocorrPlot = () => {
  const pathToData = "./mcmc_autocorr.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />;
};
