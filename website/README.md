# Website

This website is built using [Docusaurus 2](https://v2.docusaurus.io/), a modern static website generator. Ensure you have [built the Sphinx docs](https://github.com/facebookresearch/beanmachine/tree/main/sphinx) before building the website.

### Installation

```
$ yarn
```

### Local Development

```
$ export NODE_OPTIONS=--openssl-legacy-provider
$ yarn start
```

This command starts a local development server and open up a browser window. Most changes are reflected live without having to restart the server.

The first line setting the environment variable is required to bypass a bug in OpenSSL on some platforms and is likely to be removed in future versions.

### Build

```
$ export NODE_OPTIONS=--openssl-legacy-provider
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

```
$ GIT_USER=<Your GitHub username> USE_SSH=true yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

### Continuous Integration

Some common defaults for linting/formatting have been set for you. If you integrate your project with an open source Continuous Integration system (e.g. Travis CI, CircleCI), you may check for issues using the following command.

```
$ yarn ci
```
