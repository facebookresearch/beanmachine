"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[1632],{3905:function(e,t,r){r.r(t),r.d(t,{MDXContext:function(){return l},MDXProvider:function(){return f},mdx:function(){return h},useMDXComponents:function(){return u},withMDXComponents:function(){return p}});var n=r(67294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(){return i=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var r=arguments[t];for(var n in r)Object.prototype.hasOwnProperty.call(r,n)&&(e[n]=r[n])}return e},i.apply(this,arguments)}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function c(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var l=n.createContext({}),p=function(e){return function(t){var r=u(t.components);return n.createElement(e,i({},t,{components:r}))}},u=function(e){var t=n.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):c(c({},t),e)),r},f=function(e){var t=u(e.components);return n.createElement(l.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,i=e.originalType,o=e.parentName,l=s(e,["components","mdxType","originalType","parentName"]),p=u(r),f=a,m=p["".concat(o,".").concat(f)]||p[f]||d[f]||i;return r?n.createElement(m,c(c({ref:t},l),{},{components:r})):n.createElement(m,c({ref:t},l))}));function h(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=r.length,o=new Array(i);o[0]=m;var c={};for(var s in t)hasOwnProperty.call(t,s)&&(c[s]=t[s]);c.originalType=e,c.mdxType="string"==typeof e?e:a,o[1]=c;for(var l=2;l<i;l++)o[l]=r[l];return n.createElement.apply(null,o)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},96180:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return c},contentTitle:function(){return s},metadata:function(){return l},toc:function(){return p},default:function(){return f}});var n=r(87462),a=r(63366),i=(r(67294),r(3905)),o=(r(44996),["components"]),c={slug:"packages",title:"Hierarchical Mixed Effects",sidebar_label:"Hierarchical Mixed Effects (HME)"},s=void 0,l={unversionedId:"overview/packages/packages",id:"overview/packages/packages",title:"Hierarchical Mixed Effects",description:"Packages in Bean Machine let a user reuse tested, proven code for specific purposes, relieving a user from needing to write their own custom Bean Machine logic.",source:"@site/../docs/overview/packages/packages.md",sourceDirName:"overview/packages",slug:"/overview/packages/packages",permalink:"/docs/overview/packages/packages",editUrl:"https://github.com/facebookresearch/beanmachine/edit/main/website/../docs/overview/packages/packages.md",tags:[],version:"current",frontMatter:{slug:"packages",title:"Hierarchical Mixed Effects",sidebar_label:"Hierarchical Mixed Effects (HME)"}},p=[{value:"Hierarchical Mixed Effects (HME)",id:"hierarchical-mixed-effects-hme",children:[{value:"Fitting HME Models With Fixed+Random Effects and Flexible Priors",id:"fitting-hme-models-with-fixedrandom-effects-and-flexible-priors",children:[],level:3},{value:"Bean Machine Graph For Faster Performance",id:"bean-machine-graph-for-faster-performance",children:[],level:3}],level:2}],u={toc:p};function f(e){var t=e.components,r=(0,a.Z)(e,o);return(0,i.mdx)("wrapper",(0,n.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,i.mdx)("p",null,"Packages in Bean Machine let a user reuse tested, proven code for specific purposes, relieving a user from needing to write their own custom Bean Machine logic."),(0,i.mdx)("p",null,"Currently we have just one package, HME, but we encourage pull requests to add additional packages and we plan on adding additional packages as well, e.g., Gaussian Processes, in the future."),(0,i.mdx)("h2",{id:"hierarchical-mixed-effects-hme"},"Hierarchical Mixed Effects (HME)"),(0,i.mdx)("p",null,"Hierarchical mixed effects (HME) models are frequently used in Bayesian Statistics."),(0,i.mdx)("p",null,"We created the HME Python package to make our current products\u2019 code bases easier to maintain, make future statistical/ML work more efficient, and most importantly to ensure our HME methodology can be easily reused. The HME package will make hierarchical mixed effects methods widely accessible to the broader open-source community using Bean Machine."),(0,i.mdx)("h3",{id:"fitting-hme-models-with-fixedrandom-effects-and-flexible-priors"},"Fitting HME Models With Fixed+Random Effects and Flexible Priors"),(0,i.mdx)("p",null,"This release is the first version of our HME Python package. The package is capable of fitting Bayesian hierarchical mixed effects models with:"),(0,i.mdx)("ul",null,(0,i.mdx)("li",{parentName:"ul"},"any arbitrary fixed and random effects, and"),(0,i.mdx)("li",{parentName:"ul"},"it will allow users to flexibly specify priors as they wish.")),(0,i.mdx)("h3",{id:"bean-machine-graph-for-faster-performance"},"Bean Machine Graph For Faster Performance"),(0,i.mdx)("p",null,"To fit hierarchical models, HME uses MCMC (Markov chain Monte Carlo) inference techniques powered by Bean Machine Graph (BMG), which runs critical pieces of code in C++ rather than Python, to speed up the inference process significantly."),(0,i.mdx)("hr",null),(0,i.mdx)("p",null,"Facebook specific:"),(0,i.mdx)("p",null," These models are also frequently used at Facebook including Team Power and Metric Ranking products (",(0,i.mdx)("a",{parentName:"p",href:"https://fb.workplace.com/notes/418250526036381"},"https://fb.workplace.com/notes/418250526036381"),") as well as new pilot studies on ",(0,i.mdx)("a",{parentName:"p",href:"https://fb.quip.com/GxwQAIscFRz8"},"https://fb.quip.com/GxwQAIscFRz8")," and ",(0,i.mdx)("a",{parentName:"p",href:"https://fb.quip.com/UMmcAr2zczbc"},"https://fb.quip.com/UMmcAr2zczbc"),". Additionally, the Probabilistic Programming Languages (",(0,i.mdx)("a",{parentName:"p",href:"https://www.internalfb.com/intern/bunny/?q=group%20pplxfn"},"https://www.internalfb.com/intern/bunny/?q=group%20pplxfn"),") (PPL) team has collected a list of ",(0,i.mdx)("a",{parentName:"p",href:"https://fb.quip.com/rrMAAuk02Jqa"},"https://fb.quip.com/rrMAAuk02Jqa")," who can benefit from our HME methodology."),(0,i.mdx)("p",null,"BMG: ",(0,i.mdx)("a",{parentName:"p",href:"https://fb.quip.com/TDA7AIjRmScW"},"https://fb.quip.com/TDA7AIjRmScW")),(0,i.mdx)("p",null,"Ignore--saved for formatting tips:\nLet's quickly translate the model we discussed in ",(0,i.mdx)("a",{parentName:"p",href:"/docs/why_bean_machine"},'"Why Bean Machine?"')," into Bean Machine code! Although this will get you up-and-running, ",(0,i.mdx)("strong",{parentName:"p"},"it's important that you read through all of the pages in the Overview to have a complete understanding of Bean Machine"),". Happy modeling!"))}f.isMDXComponent=!0}}]);