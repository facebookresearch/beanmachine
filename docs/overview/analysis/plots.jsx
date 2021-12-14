import React from "react";
import BrowserOnly from "@docusaurus/BrowserOnly";
import useBaseUrl from "@docusaurus/useBaseUrl";

const BokehFigure = React.memo(({ pathToData }) => {
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const targetID = plotData["target_id"];
  return (
    <div className="bk-root" id={targetID}>
      <BrowserOnly fallback={<div>loading...</div>}>
        {() => {
          {
            window.Bokeh.embed.embed_item(plotData, targetID);
          }
        }}
      </BrowserOnly>
    </div>
  );
});

export const PosteriorRateDynamicPlot = () => {
  return <BokehFigure pathToData={"./posterior_rate_dynamic.json"} />;
};

export const MCMCTracePlot = () => {
  return <BokehFigure pathToData={"./mcmc_trace.json"} />;
};

export const MCMCAutocorrPlot = () => {
  return <BokehFigure pathToData={"./mcmc_autocorr.json"} />;
};
