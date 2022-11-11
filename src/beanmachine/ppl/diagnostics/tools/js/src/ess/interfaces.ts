/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {Figure} from '@bokehjs/api/plotting';
import {Legend} from '@bokehjs/models/annotations/legend';
import {Circle} from '@bokehjs/models/glyphs/circle';
import {Line} from '@bokehjs/models/glyphs/line';
import {ColumnDataSource} from '@bokehjs/models/sources/column_data_source';
import {HoverTool} from '@bokehjs/models/tools/inspectors/hover_tool';
import {Select} from '@bokehjs/models/widgets/selectbox';

// NOTE: In the corresponding Python typing files for the diagnostic tool, we define
//       similar types using a TypedDict object. TypeScript allows us to maintain
//       semantic information about the key names and their types in the same object and
//       still have the ability to loop over objects as long as we have the
//       [key: string]: any; indicator in the interface definition. This boils down to a
//       Python type of Dict[Any, Any], which again loses all type information about the
//       object we are defining. We are mirroring what is done in Python here, so we
//       keep the semantic information here at the expense of losing type information
//       similarly to what is done in Python.

export interface Data {
  ess: {
    [key: string]: {x: number[]; y: number[]; text?: string[]};
    bulk: {x: number[]; y: number[]};
    tail: {x: number[]; y: number[]};
    ruleOfThumb: {x: number[]; y: number[]; text: string[]};
  };
}

export interface Sources {
  ess: {
    [key: string]: {
      [key: string]: ColumnDataSource | undefined;
      line: ColumnDataSource;
      circle?: ColumnDataSource;
    };
    bulk: {circle: ColumnDataSource; line: ColumnDataSource};
    tail: {circle: ColumnDataSource; line: ColumnDataSource};
    ruleOfThumb: {line: ColumnDataSource};
  };
}

export interface Figures {
  ess: Figure;
}

export interface Glyphs {
  ess: {
    bulk: {circle: {glyph: Circle; hoverGlyph: Circle}; line: {glyph: Line}};
    tail: {circle: {glyph: Circle; hoverGlyph: Circle}; line: {glyph: Line}};
    ruleOfThumb: {line: {glyph: Line; hoverGlyph: Line}};
  };
}

export interface Annotations {
  ess: Legend;
}

export interface Tooltips {
  ess: {bulk: HoverTool; tail: HoverTool; ruleOfThumb: HoverTool};
}

export interface Widgets {
  rvSelect: Select;
}
