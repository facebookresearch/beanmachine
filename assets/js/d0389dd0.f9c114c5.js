"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[7824],{3905:function(e,n,t){t.r(n),t.d(n,{MDXContext:function(){return p},MDXProvider:function(){return m},mdx:function(){return h},useMDXComponents:function(){return c},withMDXComponents:function(){return u}});var r=t(67294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(){return i=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var t=arguments[n];for(var r in t)Object.prototype.hasOwnProperty.call(t,r)&&(e[r]=t[r])}return e},i.apply(this,arguments)}function o(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?o(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function s(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var p=r.createContext({}),u=function(e){return function(n){var t=c(n.components);return r.createElement(e,i({},n,{components:t}))}},c=function(e){var n=r.useContext(p),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},m=function(e){var n=c(e.components);return r.createElement(p.Provider,{value:n},e.children)},d={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},f=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,i=e.originalType,o=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),u=c(t),m=a,f=u["".concat(o,".").concat(m)]||u[m]||d[m]||i;return t?r.createElement(f,l(l({ref:n},p),{},{components:t})):r.createElement(f,l({ref:n},p))}));function h(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var i=t.length,o=new Array(i);o[0]=f;var l={};for(var s in n)hasOwnProperty.call(n,s)&&(l[s]=n[s]);l.originalType=e,l.mdxType="string"==typeof e?e:a,o[1]=l;for(var p=2;p<i;p++)o[p]=t[p];return r.createElement.apply(null,o)}return r.createElement.apply(null,t)}f.displayName="MDXCreateElement"},37670:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return l},contentTitle:function(){return s},metadata:function(){return p},toc:function(){return u},default:function(){return m}});var r=t(87462),a=t(63366),i=(t(67294),t(3905)),o=["components"],l={id:"beanstalk",title:"The Beanstalk Compiler",sidebar_label:"Compiler"},s=void 0,p={unversionedId:"overview/beanstalk/beanstalk",id:"overview/beanstalk/beanstalk",isDocsHomePage:!1,title:"The Beanstalk Compiler",description:"This page is Work in Progress!",source:"@site/../docs/overview/beanstalk/beanstalk.md",sourceDirName:"overview/beanstalk",slug:"/overview/beanstalk/beanstalk",permalink:"/docs/overview/beanstalk/beanstalk",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/../docs/overview/beanstalk/beanstalk.md",tags:[],version:"current",frontMatter:{id:"beanstalk",title:"The Beanstalk Compiler",sidebar_label:"Compiler"},sidebar:"someSidebar",previous:{title:"C++ Runtime",permalink:"/docs/overview/bmg/bmg"},next:{title:"Jupyter Notebooks",permalink:"/docs/tutorials"}},u=[{value:"Beanstalk uses the Bean Machine Graph (BMG) library",id:"beanstalk-uses-the-bean-machine-graph-bmg-library",children:[],level:3}],c={toc:u};function m(e){var n=e.components,t=(0,a.Z)(e,o);return(0,i.mdx)("wrapper",(0,r.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,i.mdx)("p",null,(0,i.mdx)("em",{parentName:"p"},"This page is Work in Progress!")),(0,i.mdx)("p",null,"Beanstalk is an experimental, just-in-time (JIT) compiler for Bean Machine. While we expect to continue to develop this compiler in the near future, currently it handles only a subset of the Bean Machine language. For example, it supports the following tutorials:"),(0,i.mdx)("ul",null,(0,i.mdx)("li",{parentName:"ul"},"Linear Regression,"),(0,i.mdx)("li",{parentName:"ul"},"Gaussian Mixture Model (1D, mixture of 2) - TODO: The currently included tutorial is more general than that,"),(0,i.mdx)("li",{parentName:"ul"},"Neal's funnel.")),(0,i.mdx)("p",null,"The subset currently is limited to:"),(0,i.mdx)("ul",null,(0,i.mdx)("li",{parentName:"ul"},"Univariate distributions,"),(0,i.mdx)("li",{parentName:"ul"},"Simple uses of tensors, for example, tensor addition and multiplication,"),(0,i.mdx)("li",{parentName:"ul"},"Limited control flow is supported,"),(0,i.mdx)("li",{parentName:"ul"},"Inference algorithms - currently only Newtonian Monte Carlo (NMC) is supported,"),(0,i.mdx)("li",{parentName:"ul"},"Only one chain of samples can be generated at a time.")),(0,i.mdx)("p",null,"To use Beanstalk to run an inference model, instead of using a standard Bean Machine inference algorithm using a command such as ",(0,i.mdx)("inlineCode",{parentName:"p"},"bm.SingleSiteNewtonianMonteCarlo().infer()"),", simply include the compiler using ",(0,i.mdx)("inlineCode",{parentName:"p"},"from beanmachine.ppl.inference.bmg_inference import BMGInference")," and use ",(0,i.mdx)("inlineCode",{parentName:"p"},"BMGInference().infer()"),"."),(0,i.mdx)("p",null,"The ",(0,i.mdx)("inlineCode",{parentName:"p"},"BMGInference()")," object provides a collection of utility methods that can be used to inspect the intermediate results of the compiler, namely:"),(0,i.mdx)("ul",null,(0,i.mdx)("li",{parentName:"ul"},(0,i.mdx)("inlineCode",{parentName:"li"},"BMGInference().infer(queries, observations, num_samples, num_chains)")," - Returns a dictionary of samples for the queried variables,"),(0,i.mdx)("li",{parentName:"ul"},(0,i.mdx)("inlineCode",{parentName:"li"},"BMGInference().to_graphviz(queries, observations)")," - Returns a graphviz graph representing the model,"),(0,i.mdx)("li",{parentName:"ul"},(0,i.mdx)("inlineCode",{parentName:"li"},"BMGInference().to_dot(queries, observations)")," - Returns a DOT representation of the probabilistic graph of the model,"),(0,i.mdx)("li",{parentName:"ul"},(0,i.mdx)("inlineCode",{parentName:"li"},"BMGInference().to_cpp(queries, observations)")," - Returns a C++ program that builds a version of this graph, and"),(0,i.mdx)("li",{parentName:"ul"},(0,i.mdx)("inlineCode",{parentName:"li"},"BMGInference().to_python(queries, observations)")," - Returns a Python program that builds a version of the graph.")),(0,i.mdx)("h3",{id:"beanstalk-uses-the-bean-machine-graph-bmg-library"},"Beanstalk uses the Bean Machine Graph (BMG) library"),(0,i.mdx)("p",null,"With code generated that is powered by the Bean Machine Graph (BMG) library, which runs critical pieces of code in C++ rather than Python, to speed up the inference process significantly."),(0,i.mdx)("hr",null),(0,i.mdx)("p",null,"Facebook specific:"),(0,i.mdx)("p",null," These models are also frequently used at Facebook including Team Power and Metric Ranking products (",(0,i.mdx)("a",{parentName:"p",href:"https://fb.workplace.com/notes/418250526036381"},"https://fb.workplace.com/notes/418250526036381"),") as well as new pilot studies on ",(0,i.mdx)("a",{parentName:"p",href:"https://fb.quip.com/GxwQAIscFRz8"},"https://fb.quip.com/GxwQAIscFRz8")," and ",(0,i.mdx)("a",{parentName:"p",href:"https://fb.quip.com/UMmcAr2zczbc"},"https://fb.quip.com/UMmcAr2zczbc"),". Additionally, the Probabilistic Programming Languages (",(0,i.mdx)("a",{parentName:"p",href:"https://www.internalfb.com/intern/bunny/?q=group%20pplxfn"},"https://www.internalfb.com/intern/bunny/?q=group%20pplxfn"),") (PPL) team has collected a list of ",(0,i.mdx)("a",{parentName:"p",href:"https://fb.quip.com/rrMAAuk02Jqa"},"https://fb.quip.com/rrMAAuk02Jqa")," who can benefit from our HME methodology."),(0,i.mdx)("p",null,"BMG: ",(0,i.mdx)("a",{parentName:"p",href:"https://fb.quip.com/TDA7AIjRmScW"},"https://fb.quip.com/TDA7AIjRmScW")),(0,i.mdx)("p",null,"Ignore--saved for formatting tips:\nLet's quickly translate the model we discussed in the ",(0,i.mdx)("a",{parentName:"p",href:"/docs/introduction"},"Introduction")," into Bean Machine code! Although this will get you up-and-running, ",(0,i.mdx)("strong",{parentName:"p"},"it's important that you read through all of the pages in the Overview to have a complete understanding of Bean Machine"),". Happy modeling!"))}m.isMDXComponent=!0}}]);