import React from 'react';
import Loadable from 'react-loadable';
import BrowserOnly from '@docusaurus/BrowserOnly';

export const BokehFigure = React.memo(({data}) => {
  const targetId = data['target_id'];
  return (
    <div
      className="bk-root thin-scrollbar"
      id={targetId}
      style={{overflow: 'auto', width: '100%'}}>
      <BrowserOnly fallback={<div>loading...</div>}>
        {() => {
          {
            window.Bokeh.embed.embed_item(data, targetId);
          }
        }}
      </BrowserOnly>
    </div>
  );
});

const Plotly = Loadable({
  loader: () => import(`react-plotly.js`),
  loading: ({timedOut}) =>
    timedOut ? (
      <blockquote>Error: Loading Plotly timed out.</blockquote>
    ) : (
      <div>phooey</div>
    ),
  timeout: 10000,
});

export const PlotlyFigure = React.memo(({data, layout}) => {
  return (
    <div className="plotly-figure">
      <Plotly data={data} layout={layout} />
    </div>
  );
});
