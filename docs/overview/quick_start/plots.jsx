import React from "react";
import BrowserOnly from "@docusaurus/BrowserOnly";
import useBaseUrl from "@docusaurus/useBaseUrl";

const BokehFigure = React.memo(({ pathToData }) => {
  const plotData = React.useMemo(() => require(`${pathToData}`), []);
  const targetID = plotData["target_id"];
  return (
    <div className="bk-root" id={targetID}>
      <BrowserOnly fallback={<img src={useBaseUrl(`/img/${targetID}.png`)} />}>
        {() => {
          {
            window.Bokeh.embed.embed_item(plotData, targetID);
          }
        }}
      </BrowserOnly>
    </div>
  );
});

export const PriorPoissonPlot = () => {
  return <BokehFigure pathToData={"./prior_poisson.json"} />;
};

export const PriorExponentialPlot = () => {
  return <BokehFigure pathToData={"./prior_exponential.json"} />;
};

export const PosteriorRateStaticPlot = () => {
  return <BokehFigure pathToData={"./posterior_rate_static.json"} />;
};
