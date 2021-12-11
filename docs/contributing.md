---
id: contributing
title: Contributing Docs
sidebar_label: Contributing Docs
---

This document describes how to add and update markdown content that is presented in this Docusaurus 2 project.

The static documentation content displayed on this site can be found in the `/docs` folder within the project itself. The folder contains markdown files which are rendered in the Docusaurus 2 UI.

The markdown files used in this project need to contain a header with additional information about the file. You can see an example header here:

```
---
id: toc
title: Table of Contents
sidebar_label: TOC
---
```


Links to other documents within the `docs` folder should contain the path to the document. For example the following links to the `why_bean_machine.md` file in the `why_bean_machine` folder under `docs`.


```
[Why Bean Machine?](_overview/why_bean_machine/why_bean_machine.md_)
```

For a full list of formatting syntax supported by Docusarus, you can visit this [site](https://www.markdownguide.org/basic-syntax/).

Docusaurus 2 also supports MDX, a format that lets you seamlessly write JSX in your Markdown documents. You can find out more by visiting [MDX](https://mdxjs.com/), and you can see an example in this project [here](mdx.md).

## Adding Your Page to the Left Navigation

To add a new page to the left navigation, you can add it to the `sidebars.js` file in the `/website` folder in the project. Details for customizing this sidebar file can be found [here](https://v2.docusaurus.io/docs/docs-introduction/#sidebar-object).

```
module.exports = {
  someSidebar: {
    Documentation: ['toc', 'overview/introduction/introduction',
    'overview/quick_start/quick_start', 'overview/modeling/modeling',
    'overview/inference/inference', 'overview/analysis/analysis']
  },
};
```
