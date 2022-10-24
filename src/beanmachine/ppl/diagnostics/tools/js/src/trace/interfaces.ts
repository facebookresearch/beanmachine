/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {ColumnDataSource} from '@bokehjs/models/sources/column_data_source';
import {HoverTool} from '@bokehjs/models/tools/inspectors/hover_tool';
import {Plot} from '@bokehjs/models/plots/plot';
import {RankHistogram} from '../stats/histogram';

// NOTE: In the corresponding Python typing files for the diagnostic tool, we define
//       similar types using a TypedDict object. TypeScript allows us to maintain
//       semantic information about the key names and their types in the same object and
//       still have the ability to loop over objects as long as we have the
//       [key: string]: any; indicator in the interface definition. This boils down to a
//       Python type of Dict[Any, Any], which again loses all type information about the
//       object we are defining. We are mirroring what is done in Python here, so we
//       keep the semantic information here at the expense of losing type information
//       similarly to what is done in Python.

export interface LineOrCircleGlyphData {
  [key: string]: any;
  x: number[];
  y: number[];
}

export interface MarginalDataSingleChain {
  [key: string]: any;
  line: LineOrCircleGlyphData;
  chain: number;
  mean: number;
  bandwidth: number;
}

export interface ForestDataSingleChain {
  [key: string]: any;
  line: LineOrCircleGlyphData;
  circle: LineOrCircleGlyphData;
  chain: number;
  mean: number;
}

export interface TraceDataSingleChain {
  [key: string]: any;
  line: LineOrCircleGlyphData;
  chain: number;
  mean: number;
}

export interface MarginalDataAllChains {
  [key: string]: MarginalDataSingleChain;
}

export interface ForestDataAllChains {
  [key: string]: ForestDataSingleChain;
}

export interface TraceDataAllChains {
  [key: string]: TraceDataSingleChain;
}

export interface Data {
  [key: string]: any;
  marginals: MarginalDataAllChains;
  forests: ForestDataAllChains;
  traces: TraceDataAllChains;
  ranks: RankHistogram;
}

export interface SourceSingleChain {
  line: ColumnDataSource;
  circle?: ColumnDataSource;
  quad?: ColumnDataSource;
}

export interface SourceAllChains {
  [key: string]: SourceSingleChain;
}

export interface Sources {
  [key: string]: any;
  marginals: SourceAllChains;
  forests: SourceAllChains;
  traces: SourceAllChains;
  ranks: SourceAllChains;
}

export interface Figures {
  [key: string]: any;
  marginals: Plot;
  forests: Plot;
  traces: Plot;
  ranks: Plot;
}

export interface RankTooltips {
  [key: string]: HoverTool;
  line: HoverTool;
  quad: HoverTool;
}

export interface Tooltips {
  [key: string]: any;
  marginals: HoverTool[];
  forests: HoverTool[];
  traces: HoverTool[];
  ranks: RankTooltips[];
}
