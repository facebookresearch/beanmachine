import React from 'react';
import { BokehFigure } from '../../../../website/src/components/Plotting.jsx';

export const SamplesDiagnosticPlots = () => {
  const pathToData = "./assets/plot_data/samples-diagnostic-plots.json";
  const data = React.useMemo(() => require(`${pathToData}`), []);
  return <BokehFigure data={data} />
};