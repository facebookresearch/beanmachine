/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {Plot} from '@bokehjs/models/plots/plot';
import {ColumnDataSource} from '@bokehjs/models/sources/column_data_source';
import {HoverTool} from '@bokehjs/models/tools/inspectors/hover_tool';
import {Div} from '@bokehjs/models/widgets/div';
import {RadioButtonGroup} from '@bokehjs/models/widgets/radio_button_group';
import {Select} from '@bokehjs/models/widgets/selectbox';
import {Slider} from '@bokehjs/models/widgets/slider';

// NOTE: In the corresponding Python typing files for the diagnostic tool, we define
//       similar types using a TypedDict object. TypeScript allows us to maintain
//       semantic information about the key names and their types in the same object and
//       still have the ability to loop over objects as long as we have the
//       [key: string]: any; indicator in the interface definition. This boils down to a
//       Python type of Dict[Any, Any], which again loses all type information about the
//       object we are defining. We are mirroring what is done in Python here, so we
//       keep the semantic information here at the expense of losing type information
//       similarly to what is done in Python.

export interface DistributionData {
  [key: string]: any;
  x: number[];
  y: number[];
  bandwidth: number;
}

export interface HDIData {
  [key: string]: any;
  base: number[];
  lower: number[];
  upper: number[];
}

export interface StatsData {
  [key: string]: any;
  x: number[];
  y: number[];
  text: string[];
}

export interface LabelsData {
  [key: string]: any;
  x: number[];
  y: number[];
  text: string[];
  text_align: string[];
  x_offset: number[];
  y_offset: number[];
}

export interface GlyphData {
  [key: string]: any;
  distribution: DistributionData;
  hdi: HDIData;
  stats: StatsData;
  labels: LabelsData;
}

export interface Data {
  [key: string]: any;
  marginal: GlyphData;
  cumulative: GlyphData;
}

export interface Source {
  [key: string]: any;
  distribution: ColumnDataSource;
  hdi: ColumnDataSource;
  stats: ColumnDataSource;
  labels: ColumnDataSource;
}

export interface Sources {
  [key: string]: any;
  marginal: Source;
  cumulative: Source;
}

export interface Figures {
  [key: string]: any;
  marginal: Plot;
  cumulative: Plot;
}

export interface Tooltip {
  [key: string]: any;
  distribution: HoverTool;
  stats: HoverTool;
}

export interface Tooltips {
  [key: string]: any;
  marginal: Tooltip;
  cumulative: Tooltip;
}

export interface Widgets {
  rv_select: Select;
  bw_factor_slider: Slider;
  bw_div: Div;
  hdi_slider: Slider;
  stats_button: RadioButtonGroup;
}
