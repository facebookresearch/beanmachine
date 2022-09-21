// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

const OFF = 0;
const WARNING = 1;
const ERROR = 2;

module.exports = {
  root: true,
  env: {
    browser: true,
    commonjs: true,
    jest: true,
    node: true,
  },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    allowImportExportEverywhere: true,
  },
  extends: ['airbnb', 'prettier', 'plugin:import/typescript'],
  plugins: ['prefer-arrow'],
  rules: {
    // Allow more than 1 class per file.
    'max-classes-per-file': ['error', {ignoreExpressions: true, max: 2}],
    // Allow snake_case.
    camelcase: [
      OFF,
      {
        properties: 'never',
        ignoreDestructuring: true,
        ignoreImports: true,
        ignoreGlobals: true,
      },
    ],
    'no-underscore-dangle': OFF,
    // Arrow function rules.
    'prefer-arrow/prefer-arrow-functions': [
      ERROR,
      {
        disallowPrototype: true,
        singleReturnOnly: false,
        classPropertiesAllowed: false,
      },
    ],
    'prefer-arrow-callback': [ERROR, {allowNamedFunctions: true}],
    'arrow-parens': [ERROR, 'always'],
    'arrow-body-style': [ERROR, 'always'],
    'func-style': [ERROR, 'declaration', {allowArrowFunctions: true}],
    'react/function-component-definition': [
      ERROR,
      {
        namedComponents: 'arrow-function',
        unnamedComponents: 'arrow-function',
      },
    ],
    // Ignore the global require, since some required packages are BrowserOnly.
    'global-require': 0,
    // We reassign several parameter objects since Bokeh is just updating values in the
    // them.
    'no-param-reassign': 0,
    // Ignore certain webpack alias because it can't be resolved
    'import/no-unresolved': [
      ERROR,
      {ignore: ['^@theme', '^@docusaurus', '^@generated', '^@bokeh']},
    ],
    'import/extensions': OFF,
    'object-shorthand': [ERROR, 'never'],
    'prefer-destructuring': [WARNING, {object: true, array: true}],
    'no-nested-ternary': 0,
  },
};
