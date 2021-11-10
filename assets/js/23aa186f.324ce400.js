"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[4819],{3905:function(e,t,n){n.r(t),n.d(t,{MDXContext:function(){return m},MDXProvider:function(){return u},mdx:function(){return f},useMDXComponents:function(){return p},withMDXComponents:function(){return d}});var a=n(67294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(){return o=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var a in n)Object.prototype.hasOwnProperty.call(n,a)&&(e[a]=n[a])}return e},o.apply(this,arguments)}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var m=a.createContext({}),d=function(e){return function(t){var n=p(t.components);return a.createElement(e,o({},t,{components:n}))}},p=function(e){var t=a.useContext(m),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},u=function(e){var t=p(e.components);return a.createElement(m.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},h=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,o=e.originalType,i=e.parentName,m=l(e,["components","mdxType","originalType","parentName"]),d=p(n),u=r,h=d["".concat(i,".").concat(u)]||d[u]||c[u]||o;return n?a.createElement(h,s(s({ref:t},m),{},{components:n})):a.createElement(h,s({ref:t},m))}));function f(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var o=n.length,i=new Array(o);i[0]=h;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:r,i[1]=s;for(var m=2;m<o;m++)i[m]=n[m];return a.createElement.apply(null,i)}return a.createElement.apply(null,n)}h.displayName="MDXCreateElement"},13919:function(e,t,n){function a(e){return!0===/^(\w*:|\/\/)/.test(e)}function r(e){return void 0!==e&&!a(e)}n.d(t,{b:function(){return a},Z:function(){return r}})},44996:function(e,t,n){n.r(t),n.d(t,{useBaseUrlUtils:function(){return o},default:function(){return i}});var a=n(52263),r=n(13919);function o(){var e=(0,a.default)().siteConfig,t=(e=void 0===e?{}:e).baseUrl,n=void 0===t?"/":t,o=e.url;return{withBaseUrl:function(e,t){return function(e,t,n,a){var o=void 0===a?{}:a,i=o.forcePrependBaseUrl,s=void 0!==i&&i,l=o.absolute,m=void 0!==l&&l;if(!n)return n;if(n.startsWith("#"))return n;if((0,r.b)(n))return n;if(s)return t+n;var d=n.startsWith(t)?n:t+n.replace(/^\//,"");return m?e+d:d}(o,n,e,t)}}}function i(e,t){return void 0===t&&(t={}),(0,o().withBaseUrl)(e,t)}},99423:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return m},metadata:function(){return d},toc:function(){return p},default:function(){return c}});var a=n(87462),r=n(63366),o=(n(67294),n(3905)),i=n(44996),s=["components"],l={id:"quick_start",title:"Quick Start",sidebar_label:"Quick Start",slug:"/quickstart"},m=void 0,d={unversionedId:"overview/quick_start/quick_start",id:"overview/quick_start/quick_start",isDocsHomePage:!1,title:"Quick Start",description:"Let's quickly translate the model we discussed in the Introduction into Bean Machine code! Although this will get you up-and-running, it's important that you read through all of the pages in the Overview to have a complete understanding of Bean Machine. Happy modeling!",source:"@site/../docs/overview/quick_start/quick_start.md",sourceDirName:"overview/quick_start",slug:"/quickstart",permalink:"/docs/quickstart",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/../docs/overview/quick_start/quick_start.md",tags:[],version:"current",frontMatter:{id:"quick_start",title:"Quick Start",sidebar_label:"Quick Start",slug:"/quickstart"},sidebar:"someSidebar",previous:{title:"Introduction",permalink:"/docs/introduction"},next:{title:"Modeling",permalink:"/docs/overview/modeling/modeling"}},p=[{value:"Modeling",id:"modeling",children:[],level:2},{value:"Data",id:"data",children:[],level:2},{value:"Inference",id:"inference",children:[],level:2},{value:"Analysis",id:"analysis",children:[],level:2},{value:"We&#39;re not done yet!",id:"were-not-done-yet",children:[],level:2}],u={toc:p};function c(e){var t=e.components,n=(0,r.Z)(e,s);return(0,o.mdx)("wrapper",(0,a.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,o.mdx)("p",null,"Let's quickly translate the model we discussed in the ",(0,o.mdx)("a",{parentName:"p",href:"/docs/introduction"},"Introduction")," into Bean Machine code! Although this will get you up-and-running, ",(0,o.mdx)("strong",{parentName:"p"},"it's important that you read through all of the pages in the Overview to have a complete understanding of Bean Machine"),". Happy modeling!"),(0,o.mdx)("h2",{id:"modeling"},"Modeling"),(0,o.mdx)("p",null,"As a quick refresher, we're writing a model to understand a disease's reproduction rate, based on the number of new cases of that disease we've seen. Though we never observe the true reproduction rate, let's start off with a prior distribution that represents our beliefs about the reproduction rate before seeing any data."),(0,o.mdx)("pre",null,(0,o.mdx)("code",{parentName:"pre",className:"language-py"},"import beanmachine.ppl as bm\nimport torch.distributions as dist\n\n@bm.random_variable\ndef reproduction_rate():\n    # Exponential distribution with rate 10 has mean 0.1.\n    return dist.Exponential(rate=10.0)\n")),(0,o.mdx)("p",null,"There are a few things to notice here!"),(0,o.mdx)("ul",null,(0,o.mdx)("li",{parentName:"ul"},"Most importantly, we've decorated this function with ",(0,o.mdx)("inlineCode",{parentName:"li"},"@bm.random_variable"),". This is how you tell Bean Machine to interpret this function probabilistically. ",(0,o.mdx)("inlineCode",{parentName:"li"},"@bm.random_variable")," functions are the building blocks of Bean Machine models, and let the framework explore different values that the function represents when fitting a good distribution for observed data that you'll provide later."),(0,o.mdx)("li",{parentName:"ul"},"Next, notice that the function returns a ",(0,o.mdx)("a",{parentName:"li",href:"https://pytorch.org/docs/stable/distributions.html?highlight=distribution#module-torch.distributions"},"PyTorch distribution"),". This distribution encodes your prior belief about a particular random variable. In the case of ",(0,o.mdx)("span",{parentName:"li",className:"math math-inline"},(0,o.mdx)("span",{parentName:"span",className:"katex"},(0,o.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,o.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,o.mdx)("semantics",{parentName:"math"},(0,o.mdx)("mrow",{parentName:"semantics"},(0,o.mdx)("mtext",{parentName:"mrow"},"Exponential"),(0,o.mdx)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,o.mdx)("mn",{parentName:"mrow"},"10.0"),(0,o.mdx)("mo",{parentName:"mrow",stretchy:"false"},")")),(0,o.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\text{Exponential}(10.0)")))),(0,o.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,o.mdx)("span",{parentName:"span",className:"base"},(0,o.mdx)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,o.mdx)("span",{parentName:"span",className:"mord text"},(0,o.mdx)("span",{parentName:"span",className:"mord"},"Exponential")),(0,o.mdx)("span",{parentName:"span",className:"mopen"},"("),(0,o.mdx)("span",{parentName:"span",className:"mord"},"1"),(0,o.mdx)("span",{parentName:"span",className:"mord"},"0"),(0,o.mdx)("span",{parentName:"span",className:"mord"},"."),(0,o.mdx)("span",{parentName:"span",className:"mord"},"0"),(0,o.mdx)("span",{parentName:"span",className:"mclose"},")"))))),", our prior has this shape:")),(0,o.mdx)("img",{src:(0,i.default)("/img/exponential_10.png")}),(0,o.mdx)("ul",null,(0,o.mdx)("li",{parentName:"ul"},"As you can see, the prior encourages smaller values for the reproduction rate, averaging at a rate of 10%, but allows for the possibility of much larger spread rates."),(0,o.mdx)("li",{parentName:"ul"},"Lastly, realize that although you've provided a prior distribution here, the framework will automatically \"refine\" this distribution, as it searches for values that represent observed data that you'll provide later. So, after we fit the model to observed data, the random variable will no longer look like the graph shown above!")),(0,o.mdx)("p",null,"The last piece of the model is how the reproduction rate relates to the new cases of illness that we observe the subsequent day. This number of new cases is related to the underlying reproduction rate -- how fast the virus tends to spread -- as well as the current number of cases. However, it's not a deterministic function of those two values. Instead, it depends on a lot of environmental factors like social behavior, stochasticity of transmission, and so on. It would be far too complicated to capture all of those factors in a single model. Instead, we'll aggregate all of these environmental factors in the form of a probability distribution, the ",(0,o.mdx)("span",{parentName:"p",className:"math math-inline"},(0,o.mdx)("span",{parentName:"span",className:"katex"},(0,o.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,o.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,o.mdx)("semantics",{parentName:"math"},(0,o.mdx)("mrow",{parentName:"semantics"},(0,o.mdx)("mtext",{parentName:"mrow"},"Poisson")),(0,o.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\text{Poisson}")))),(0,o.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,o.mdx)("span",{parentName:"span",className:"base"},(0,o.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.68333em",verticalAlign:"0em"}}),(0,o.mdx)("span",{parentName:"span",className:"mord text"},(0,o.mdx)("span",{parentName:"span",className:"mord"},"Poisson"))))))," distribution."),(0,o.mdx)("p",null,"Let's say, for this example, we observed a little over a million, 1087980, cases today. We use such a precise number here to remind you that this is a known value and not a random one. In this case, if the disease were to happen to have a reproduction rate of 0.1, this is what our ",(0,o.mdx)("span",{parentName:"p",className:"math math-inline"},(0,o.mdx)("span",{parentName:"span",className:"katex"},(0,o.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,o.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,o.mdx)("semantics",{parentName:"math"},(0,o.mdx)("mrow",{parentName:"semantics"},(0,o.mdx)("mtext",{parentName:"mrow"},"Poisson")),(0,o.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\text{Poisson}")))),(0,o.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,o.mdx)("span",{parentName:"span",className:"base"},(0,o.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.68333em",verticalAlign:"0em"}}),(0,o.mdx)("span",{parentName:"span",className:"mord text"},(0,o.mdx)("span",{parentName:"span",className:"mord"},"Poisson"))))))," distribution for new cases would look like:"),(0,o.mdx)("img",{src:(0,i.default)("/img/poisson.png")}),(0,o.mdx)("p",null,"Let's write this up in Bean Machine. Using the syntax we've already seen, it's pretty simple:"),(0,o.mdx)("pre",null,(0,o.mdx)("code",{parentName:"pre",className:"language-py"},"num_infected = 1087980\n\n@bm.random_variable\ndef num_new_cases():\n    return dist.Poisson(reproduction_rate() *  num_infected)\n")),(0,o.mdx)("p",null,"As you can see, this function relies on the ",(0,o.mdx)("inlineCode",{parentName:"p"},"reproduction_rate()")," that we defined before. Do notice: even though ",(0,o.mdx)("inlineCode",{parentName:"p"},"reproduction_rate()")," returns a distribution, here the return value from ",(0,o.mdx)("inlineCode",{parentName:"p"},"reproduction_rate()")," is treated like a sample from that distribution! Bean Machine works hard behind the scenes to sample efficiently from distributions, so that you can easily build sophisticated models that only have to reason about these samples."),(0,o.mdx)("h2",{id:"data"},"Data"),(0,o.mdx)("p",null,"With the model fully defined, we should gather some data to learn about! In the real world, you might work with a government agency to determine the number of real, new cases observed on the next day. For sake of our example, let's say that we observed 238154 new cases on the next day. Bean Machine's random variable syntax allows you to bind this information directly as an observation for the ",(0,o.mdx)("inlineCode",{parentName:"p"},"num_new_cases()")," random variable within a simple Python dictionary. Here's how to do it:"),(0,o.mdx)("pre",null,(0,o.mdx)("code",{parentName:"pre",className:"language-py"},"from torch import tensor\n\nobservations = {\n    # PyTorch distributions expect tensors, so we provide a tensor here.\n    num_new_cases(): tensor(238154.),\n}\n")),(0,o.mdx)("p",null,"Using a random variable function as keys in this dictionary may feel unusual at first, but it quickly becomes an intuitive way to reference these random variable functions by name!"),(0,o.mdx)("h2",{id:"inference"},"Inference"),(0,o.mdx)("p",null,"With model and observations in hand, we're ready for the fun part: inference! Inference is the process of combining ",(0,o.mdx)("em",{parentName:"p"},"model")," with ",(0,o.mdx)("em",{parentName:"p"},"data")," to obtain ",(0,o.mdx)("em",{parentName:"p"},"insights"),", in the form of probability distributions over values of interest. Bean Machine offers a powerful and general inference framework to enable fitting arbitrary models to data."),(0,o.mdx)("p",null,"The call to inference involves first creating an appropriate inference engine object and then invoking the ",(0,o.mdx)("inlineCode",{parentName:"p"},"infer")," method:"),(0,o.mdx)("pre",null,(0,o.mdx)("code",{parentName:"pre",className:"language-py"},"samples = bm.CompositionalInference().infer(\n    queries=[ reproduction_rate() ],\n    observations=observations,\n    num_samples=10000,\n)\n")),(0,o.mdx)("p",null,"There's a lot going on here! First, let's take a look at the inference method that we used, ",(0,o.mdx)("inlineCode",{parentName:"p"},"CompositionalInference()"),". Bean Machine supports generic inference, which means that it can fit your model to the data without knowing the intricate and particular workings of the model that you defined. However, there are lots of ways of performing this, and Bean Machine supports a rich library of inference methods that can work for different kinds of models. For now, all you need to know is that ",(0,o.mdx)("inlineCode",{parentName:"p"},"CompositionalInference")," is a general inference strategy that will try to automatically determine the best inference method(s) to use for your model, based on the definitions of random variables you've provided. It should work well for this simple model. You can check out our guides on ",(0,o.mdx)("a",{parentName:"p",href:"/docs/overview/inference/inference"},"Inference")," to learn more!"),(0,o.mdx)("p",null,"Let's take a look at the parameters to ",(0,o.mdx)("inlineCode",{parentName:"p"},"infer()"),". In ",(0,o.mdx)("inlineCode",{parentName:"p"},"queries"),", you provide a list of random variables that you're interested in learning about. Bean Machine will learn probability distributions for these, and will return them to you when inference completes! Note that this uses exactly the same pattern to reference random variables that we used when binding data."),(0,o.mdx)("p",null,"We bind our real-world observations with the ",(0,o.mdx)("inlineCode",{parentName:"p"},"observations")," parameter. This provides a set of probabilistic constraints that Bean Machine seeks to satisfy during inference. In particular, Bean Machine tries to fit probability distributions for unobserved random variables, so that those probability distributions explain the observed data -- and your prior beliefs -- well."),(0,o.mdx)("p",null,"Lastly, ",(0,o.mdx)("inlineCode",{parentName:"p"},"num_samples")," is the number of samples that you want to learn. Bean Machine doesn't learn smooth probability distributions for your ",(0,o.mdx)("inlineCode",{parentName:"p"},"queries"),", but instead accumulates a representative set of samples from those distributions. This parameter lets you specify how many samples should comprise these distributions."),(0,o.mdx)("h2",{id:"analysis"},"Analysis"),(0,o.mdx)("p",null,"Our results are ready! Let's visualize results for the reproduction rate."),(0,o.mdx)("p",null,"The ",(0,o.mdx)("inlineCode",{parentName:"p"},"samples")," object that we have now contains samples from the probability distributions that we've fit for our model and data. It supports dictionary-like indexing using -- you guessed it -- the same random variable referencing syntax we've seen before. A second index (here, ",(0,o.mdx)("inlineCode",{parentName:"p"},"[0]"),") selects one of the inference chains generated by the sampling algorithm; this will be explained in the Inference section, so let us just use ",(0,o.mdx)("inlineCode",{parentName:"p"},"0")," for now."),(0,o.mdx)("pre",null,(0,o.mdx)("code",{parentName:"pre",className:"language-py"},"reproduction_rate_samples = samples[ reproduction_rate() ][0]\nreproduction_rate_samples\n")),(0,o.mdx)("pre",null,(0,o.mdx)("code",{parentName:"pre"},"tensor([0.0146, 0.1720, 0.1720,  ..., 0.2187, 0.2187, 0.2187])\n")),(0,o.mdx)("p",null,"Let's visualize that more intuitively."),(0,o.mdx)("pre",null,(0,o.mdx)("code",{parentName:"pre",className:"language-py"},'import matplotlib.pyplot as plt\n\nplt.hist(reproduction_rate_samples, label="reproduction_rate_samples")\nplt.axvline(reproduction_rate_samples.mean(), label=f"Posterior mean = {reproduction_rate_samples.mean() :.2f}", color="K")\nplt.xlabel("reproduction_rate")\nplt.ylabel("Probability density")\nplt.legend();\n')),(0,o.mdx)("img",{src:(0,i.default)("/img/results.png")}),(0,o.mdx)("p",null,"This histogram represents our beliefs over the underlying reproduction rate, after observing the current day's worth of new cases. You'll note that it balancing our prior beliefs with rate that we learn just from looking at the new data. It also captures the uncertainty inherent in our estimate!"),(0,o.mdx)("h2",{id:"were-not-done-yet"},"We're not done yet!"),(0,o.mdx)("p",null,"This is the tip of the iceberg. The rest of this ",(0,o.mdx)("strong",{parentName:"p"},"Overview")," will cover critical concepts from the above sections. Read on to learn how to make the most of Bean Machine's powerful modeling and inference systems!"))}c.isMDXComponent=!0}}]);