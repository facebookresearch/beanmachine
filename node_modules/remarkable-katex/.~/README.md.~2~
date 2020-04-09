[![Build Status](https://travis-ci.org/bradhowes/remarkable-katex.svg?branch=master)](https://travis-ci.org/bradhowes/remarkable-katex)

# Overview

This is a [Remarkable](https://github.com/jonschlinkert/remarkable) plugin that converts
[LaTeX math expressions](http://web.ift.uib.no/Teori/KURS/WRK/TeX/symALL.html) between `$...$` (inline) or
`$$...$$` (block) delimiters into math HTML. It should not interfere with any other Markdown processing.

I use this to perform server-side math expression rendering for my
blog, [Keystroke Countdown](https://keystrokecountdown.com).

> **NOTE**: currently the delimiters are hard–coded. Customizing this is work for a future release.

# To Use

Install this package using `npm`:

```bash
% npm install [-s] remarkable-katex
```

Assuming you already have `Remarkable` installed, one way to use would be like so:

```javascript
var Remarkable = require('remarkable');
var plugin = require('remarkable-katex');
var md = new Remarkable();
md.use(plugin);
```

# Configuration

None right now.

# Dependencies

* [katex](https://github.com/Khan/KaTeX) -- performs the rendering of the LaTeX commands.

# Tests

There are a set of [Vows](http://vowsjs.org) in [index.test.js](index.test.js). To run:

```bash
% npm test
```

> **NOTE**: if this fails, there may be a path issue with `vows` executable. See [package.json](package.json).
