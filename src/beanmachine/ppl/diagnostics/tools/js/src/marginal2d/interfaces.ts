/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {Figure} from '@bokehjs/api/plotting';
import {Circle, Line} from '@bokehjs/models/glyphs';
import {ColumnDataSource} from '@bokehjs/models/sources/column_data_source';
import {HoverTool} from '@bokehjs/models/tools/inspectors/hover_tool';
import {Div} from '@bokehjs/models/widgets/div';
import {Select} from '@bokehjs/models/widgets/selectbox';
import {Slider} from '@bokehjs/models/widgets/slider';

export interface XData {
  distribution: {x: number[]; y: number[]; bandwidth: number};
  hdi: {base: number[]; lower: number[]; upper: number[]};
  stats: {x: number[]; y: number[]; text: string[]};
}

export interface YData {
  distribution: {x: number[]; y: number[]; bandwidth: number};
  hdi: {
    lower: {base: number[]; lower: number[]; upper: number[]};
    upper: {base: number[]; lower: number[]; upper: number[]};
  };
  stats: {x: number[]; y: number[]; text: string[]};
}

export interface XYData {
  distribution: {x: number[]; y: number[]};
  hdi: {
    x: {
      lower: {x: number[]; y: number[]};
      upper: {x: number[]; y: number[]};
    };
    y: {
      lower: {x: number[]; y: number[]};
      upper: {x: number[]; y: number[]};
    };
  };
  stats: {x: number[]; y: number[]; text: string[]};
}

export interface Data {
  x: XData;
  y: YData;
  xy: XYData;
}

export interface Glyphs {
  x: {
    distribution: {glyph: Line; hover_glyph: Line};
    stats: {glyph: Circle; hover_glyph: Circle};
  };
  y: {
    distribution: {glyph: Line; hover_glyph: Line};
    stats: {glyph: Circle; hover_glyph: Circle};
  };
  xy: {
    distribution: Circle;
    hdi: {
      x: {
        lower: {glyph: Line; hover_glyph: Line};
        upper: {glyph: Line; hover_glyph: Line};
      };
      y: {
        lower: {glyph: Line; hover_glyph: Line};
        upper: {glyph: Line; hover_glyph: Line};
      };
    };
    stats: {glyph: Circle; hover_glyph: Circle};
  };
}

export interface Figures {
  x: Figure;
  y: Figure;
  xy: Figure;
}

export interface Sources {
  x: {distribution: ColumnDataSource; hdi: ColumnDataSource; stats: ColumnDataSource};
  y: {
    distribution: ColumnDataSource;
    hdi: {lower: ColumnDataSource; upper: ColumnDataSource};
    stats: ColumnDataSource;
  };
  xy: {
    distribution: ColumnDataSource;
    hdi: {
      x: {lower: ColumnDataSource; upper: ColumnDataSource};
      y: {lower: ColumnDataSource; upper: ColumnDataSource};
    };
    stats: ColumnDataSource;
  };
}

export interface Tooltips {
  x: {distribution: HoverTool; stats: HoverTool};
  y: {distribution: HoverTool; stats: HoverTool};
  xy: {
    distribution: HoverTool;
    hdi: {
      x: {lower: HoverTool; upper: HoverTool};
      y: {lower: HoverTool; upper: HoverTool};
    };
    stats: HoverTool;
  };
}

export interface Widgets {
  rv_select_x: Select;
  rv_select_y: Select;
  bw_factor_slider: Slider;
  hdi_slider_x: Slider;
  hdi_slider_y: Slider;
  bw_div_x: Div;
  bw_div_y: Div;
}
