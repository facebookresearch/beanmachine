---
id: adding-content
title: Adding Content
sidebar_label: Adding Content
---
### Markdown Formatting

You can add content to the documentation section by creating a new
markdown file in the website/docs folder and prepending it with a block containting the following
information:
```
---
id: some-id
title: Content Title
sidebar_label: Content Title
---
```
The rest of the document is done in standard Markdown. (See the
[Style Guide](doc1.md) for syntax.)

To preview your document, add it to the sidebars.js file in the website
folder. Here we've added the document `some-id` to the list.

```
module.exports = {
  docs: {
    BeanMachine: ['overview', 'some-id'],
    'Wiki Styles': ['doc1'],
  },
};
```
To view the rendered site, navigate to the website folder
and run `yarn start`.


### Math Content

**Example:**

Lift($L$) can be determined by Lift Coefficient ($C_L$) like the following equation.

$$
L = \frac{1}{2} \rho v^2 S C_L
$$

The syntax for the rendering equation above looks like this.

```
$$
L = \frac{1}{2} \rho v^2 S C_L
$$
```
Block rendering is enclosed in double \$ symbols \$$. Single \$ will render the math inline.
If you need to use a \$ symbol you can escape it `\$`.

The WikiBook section [LaTeX/Mathmatics](https://en.wikibooks.org/wiki/LaTeX/Mathematics),
provides a good reference for the syntax used for renderig equations.
