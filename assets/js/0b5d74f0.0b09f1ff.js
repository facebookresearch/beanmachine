"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[4165],{3905:function(e,n,t){t.r(n),t.d(n,{MDXContext:function(){return d},MDXProvider:function(){return c},mdx:function(){return f},useMDXComponents:function(){return p},withMDXComponents:function(){return l}});var a=t(67294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(){return i=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var t=arguments[n];for(var a in t)Object.prototype.hasOwnProperty.call(t,a)&&(e[a]=t[a])}return e},i.apply(this,arguments)}function o(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function s(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?o(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function m(e,n){if(null==e)return{};var t,a,r=function(e,n){if(null==e)return{};var t,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var d=a.createContext({}),l=function(e){return function(n){var t=p(n.components);return a.createElement(e,i({},n,{components:t}))}},p=function(e){var n=a.useContext(d),t=n;return e&&(t="function"==typeof e?e(n):s(s({},n),e)),t},c=function(e){var n=p(e.components);return a.createElement(d.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},h=a.forwardRef((function(e,n){var t=e.components,r=e.mdxType,i=e.originalType,o=e.parentName,d=m(e,["components","mdxType","originalType","parentName"]),l=p(t),c=r,h=l["".concat(o,".").concat(c)]||l[c]||u[c]||i;return t?a.createElement(h,s(s({ref:n},d),{},{components:t})):a.createElement(h,s({ref:n},d))}));function f(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var i=t.length,o=new Array(i);o[0]=h;var s={};for(var m in n)hasOwnProperty.call(n,m)&&(s[m]=n[m]);s.originalType=e,s.mdxType="string"==typeof e?e:r,o[1]=s;for(var d=2;d<i;d++)o[d]=t[d];return a.createElement.apply(null,o)}return a.createElement.apply(null,t)}h.displayName="MDXCreateElement"},58282:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return s},contentTitle:function(){return m},metadata:function(){return d},toc:function(){return l},default:function(){return c}});var a=t(87462),r=t(63366),i=(t(67294),t(3905)),o=["components"],s={title:"Inference",sidebar_label:"Inference",slug:"/overview/inference"},m=void 0,d={unversionedId:"overview/inference/inference",id:"overview/inference/inference",title:"Inference",description:"Inference is the process of combining a model with data to obtain insights, in the form of probability distributions over values of interest.",source:"@site/../docs/overview/inference/inference.md",sourceDirName:"overview/inference",slug:"/overview/inference",permalink:"/docs/overview/inference",editUrl:"https://github.com/facebookresearch/beanmachine/edit/main/website/../docs/overview/inference/inference.md",tags:[],version:"current",frontMatter:{title:"Inference",sidebar_label:"Inference",slug:"/overview/inference"},sidebar:"someSidebar",previous:{title:"Modeling",permalink:"/docs/overview/modeling"},next:{title:"Analysis",permalink:"/docs/overview/analysis/overview/analysis"}},l=[{value:"Prior and Posterior Distributions",id:"prior-and-posterior-distributions",children:[],level:2},{value:'<a name="binding-data"></a>Binding Data',id:"binding-data",children:[],level:2},{value:"Running Inference",id:"running-inference",children:[],level:2}],p={toc:l};function c(e){var n=e.components,t=(0,r.Z)(e,o);return(0,i.mdx)("wrapper",(0,a.Z)({},p,t,{components:n,mdxType:"MDXLayout"}),(0,i.mdx)("p",null,"Inference is the process of combining a ",(0,i.mdx)("em",{parentName:"p"},"model")," with ",(0,i.mdx)("em",{parentName:"p"},"data")," to obtain ",(0,i.mdx)("em",{parentName:"p"},"insights"),", in the form of probability distributions over values of interest."),(0,i.mdx)("p",null,"A little note on vocabulary: You've already seen in ",(0,i.mdx)("a",{parentName:"p",href:"/docs/overview/modeling"},"Modeling")," that the ",(0,i.mdx)("em",{parentName:"p"},"model")," in Bean Machine is comprised of random variable functions. In Bean Machine, the ",(0,i.mdx)("em",{parentName:"p"},"data")," is built up of a dictionary mapping random variable functions to their observed values, and ",(0,i.mdx)("em",{parentName:"p"},"insights")," take the form of discrete samples from a probability distribution. We refer to the random variables for which we're learning distributions as ",(0,i.mdx)("em",{parentName:"p"},"queried random variables"),"."),(0,i.mdx)("p",null,"Let's make this concrete by returning to our disease modeling example. As a refresher, here's the full model:"),(0,i.mdx)("pre",null,(0,i.mdx)("code",{parentName:"pre",className:"language-py"},"reproduction_rate_rate = 10.0\nnum_init = 1087980\ntime = [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)]\n\n@bm.random_variable\ndef reproduction_rate():\n    return dist.Exponential(rate=reproduction_rate_rate)\n\n@bm.functional\ndef num_total(today):\n    if today <= time[0]:\n        return num_init\n    else:\n        yesterday = today - timedelta(days=1)\n        return num_new(today) + num_total(yesterday)\n\n@bm.random_variable\ndef num_new(today):\n    yesterday = today - timedelta(days=1)\n    return dist.Poisson(reproduction_rate() * num_total(yesterday))\n")),(0,i.mdx)("h2",{id:"prior-and-posterior-distributions"},"Prior and Posterior Distributions"),(0,i.mdx)("p",null,"The ",(0,i.mdx)("span",{parentName:"p",className:"math math-inline"},(0,i.mdx)("span",{parentName:"span",className:"katex"},(0,i.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,i.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,i.mdx)("semantics",{parentName:"math"},(0,i.mdx)("mrow",{parentName:"semantics"},(0,i.mdx)("mtext",{parentName:"mrow"},"Exponential")),(0,i.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\text{Exponential}")))),(0,i.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,i.mdx)("span",{parentName:"span",className:"base"},(0,i.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.8888799999999999em",verticalAlign:"-0.19444em"}}),(0,i.mdx)("span",{parentName:"span",className:"mord text"},(0,i.mdx)("span",{parentName:"span",className:"mord"},"Exponential"))))))," distribution used here represents our beliefs about the disease's reproduction rate before seeing any data, and is known as the ",(0,i.mdx)("em",{parentName:"p"},"prior distribution"),". We've visualized this distribution previously: it represents a reproduction rate that is around 10% on average, but could be as high as 50%, and is highly right-skewed (the right side has a long tail). Values associated with prior distributions (here ",(0,i.mdx)("inlineCode",{parentName:"p"},"reproduction_rate()"),") are known as ",(0,i.mdx)("em",{parentName:"p"},"latent variables"),"."),(0,i.mdx)("p",null,"While the prior distribution encodes our prior beliefs, inference will perform the important task of adjusting latent variable values so that they balance both our prior belief and our knowledge from observed data. We refer to this distribution, after conditioning on observed data, as a ",(0,i.mdx)("em",{parentName:"p"},"posterior distribution"),". And the remaining parts of the generative model, which determine the notion of consistency used to match the latent variables with the observations, are collectively called the ",(0,i.mdx)("em",{parentName:"p"},"likelihood terms")," of the model (here consisting of ",(0,i.mdx)("inlineCode",{parentName:"p"},"num_total(today)")," and ",(0,i.mdx)("inlineCode",{parentName:"p"},"num_new(today)"),"). The way inference is performed depends upon the specific numerical method used, but it does always mean that inferred distributions will blend smoothly from resembling your prior distribution, when there is little data observed, to more wholly representing your observed data, when there are many observations."),(0,i.mdx)("h2",{id:"binding-data"},(0,i.mdx)("a",{name:"binding-data"}),"Binding Data"),(0,i.mdx)("p",null,"Inference requires us to bind data to the model in order to learn posterior distributions for our queried random variables. This is achieved by passing an ",(0,i.mdx)("inlineCode",{parentName:"p"},"observations")," dictionary to Bean Machine at inference time. Instead of sampling from random variables contained in that dictionary, Bean Machine will consider them to take on the constant values provided, and will try to find values for other random variables in your model that are consistent with the ",(0,i.mdx)("inlineCode",{parentName:"p"},"observations"),". For this example model, we can bind a few days of data as follows, taking care to match the ",(0,i.mdx)("span",{parentName:"p",className:"math math-inline"},(0,i.mdx)("span",{parentName:"span",className:"katex"},(0,i.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,i.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,i.mdx)("semantics",{parentName:"math"},(0,i.mdx)("mrow",{parentName:"semantics"},(0,i.mdx)("mtext",{parentName:"mrow"},"Poisson")),(0,i.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\text{Poisson}")))),(0,i.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,i.mdx)("span",{parentName:"span",className:"base"},(0,i.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.68333em",verticalAlign:"0em"}}),(0,i.mdx)("span",{parentName:"span",className:"mord text"},(0,i.mdx)("span",{parentName:"span",className:"mord"},"Poisson"))))))," distributions in ",(0,i.mdx)("inlineCode",{parentName:"p"},"num_new()")," with the corresponding ",(0,i.mdx)("em",{parentName:"p"},"increases")," in infection counts which they're modelling:"),(0,i.mdx)("pre",null,(0,i.mdx)("code",{parentName:"pre",className:"language-py"},"case_history = tensor([num_init, 1381734., 1630446.])\nobservations = {num_new(t): d for t, d in zip(time[1:], case_history.diff())}\n")),(0,i.mdx)("p",null,"Though correct, that code is a bit difficult to read for pedagogical purposes. The following code is equivalent:"),(0,i.mdx)("pre",null,(0,i.mdx)("code",{parentName:"pre",className:"language-py"},"observations = {\n    num_new(date(2021, 1, 2)): tensor(293754.),\n    num_new(date(2021, 1, 3)): tensor(248712.),\n}\n")),(0,i.mdx)("p",null,"Recall that calls to random variable functions from ordinary functions (including the Python toplevel) return ",(0,i.mdx)("inlineCode",{parentName:"p"},"RVIdentifier")," objects. So, the keys of this dictionary are ",(0,i.mdx)("inlineCode",{parentName:"p"},"RVIdentifiers"),", and the values are values of observed data corresponding to each key that you provide. Note that the value for a particular observation must be of the same type as the ",(0,i.mdx)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.support"},"support for the distribution that it's bound to"),". In this case, the ",(0,i.mdx)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/distributions.html#torch.distributions.poisson.Poisson.support"},"support for the ",(0,i.mdx)("span",{parentName:"a",className:"math math-inline"},(0,i.mdx)("span",{parentName:"span",className:"katex"},(0,i.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,i.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,i.mdx)("semantics",{parentName:"math"},(0,i.mdx)("mrow",{parentName:"semantics"},(0,i.mdx)("mtext",{parentName:"mrow"},"Poisson")),(0,i.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\text{Poisson}")))),(0,i.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,i.mdx)("span",{parentName:"span",className:"base"},(0,i.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.68333em",verticalAlign:"0em"}}),(0,i.mdx)("span",{parentName:"span",className:"mord text"},(0,i.mdx)("span",{parentName:"span",className:"mord"},"Poisson"))))))," distribution")," is scalar and non-negative, so what we have bound here are bounded scalar tensors."),(0,i.mdx)("h2",{id:"running-inference"},"Running Inference"),(0,i.mdx)("p",null,"We're finally ready to run inference! Let's take a look first, and then we'll explain what's happening."),(0,i.mdx)("pre",null,(0,i.mdx)("code",{parentName:"pre",className:"language-py"},"samples = bm.CompositionalInference().infer(\n    queries=[reproduction_rate()],\n    observations=observations,\n    num_samples=7000,\n    num_adaptive_samples=3000,\n    num_chains=4,\n)\n")),(0,i.mdx)("p",null,"Let's break this down. There is an inference method (in this example, that's the ",(0,i.mdx)("inlineCode",{parentName:"p"},"CompositionalInference")," class), and there's a call to ",(0,i.mdx)("inlineCode",{parentName:"p"},"infer()"),"."),(0,i.mdx)("p",null,"Inference methods are simply classes that extend from ",(0,i.mdx)("inlineCode",{parentName:"p"},"AbstractInference"),". These classes define the engine that will be used in order to fit posterior distributions to queried random variables given observations. In this particular example, we've chosen to use the specific inference method ",(0,i.mdx)("inlineCode",{parentName:"p"},"CompositionalInference")," to run inference for our disease modeling problem."),(0,i.mdx)("p",null,(0,i.mdx)("inlineCode",{parentName:"p"},"CompositionalInference")," is a powerful, flexible class for configuring inference in a variety of ways. By default, ",(0,i.mdx)("inlineCode",{parentName:"p"},"CompositionalInference")," will select an inference method for each random variable that is appropriate based on its support. For example, for differentiable random variables, this inference method will attempt to leverage gradient information when generating samples from the posterior; for discrete random variables, it will use a uniform sampler to get representative draws for each discrete value."),(0,i.mdx)("p",null,"A full discussion of the powerful ",(0,i.mdx)("inlineCode",{parentName:"p"},"CompositionalInference")," method, including extensive instructions on how to configure it to tailor specific inference methods for particular random variables, can be found in the ",(0,i.mdx)("a",{parentName:"p",href:"/docs/compositional_inference"},"Compositional Inference")," guide. Bean Machine offers a variety of other inference methods as well, which can perform differently based on the particular model you're working with. You can learn more about these inference methods under the ",(0,i.mdx)("a",{parentName:"p",href:"/docs/inference"},"Inference")," framework topic."),(0,i.mdx)("p",null,"Regardless of the inference method, ",(0,i.mdx)("inlineCode",{parentName:"p"},"infer()")," has a few important general parameters:"),(0,i.mdx)("table",null,(0,i.mdx)("thead",{parentName:"table"},(0,i.mdx)("tr",{parentName:"thead"},(0,i.mdx)("th",{parentName:"tr",align:null},"Name"),(0,i.mdx)("th",{parentName:"tr",align:null},"Usage"))),(0,i.mdx)("tbody",{parentName:"table"},(0,i.mdx)("tr",{parentName:"tbody"},(0,i.mdx)("td",{parentName:"tr",align:null},(0,i.mdx)("inlineCode",{parentName:"td"},"queries")),(0,i.mdx)("td",{parentName:"tr",align:null},"A list of random variable functions to fit posterior distributions for.")),(0,i.mdx)("tr",{parentName:"tbody"},(0,i.mdx)("td",{parentName:"tr",align:null},(0,i.mdx)("inlineCode",{parentName:"td"},"observations")),(0,i.mdx)("td",{parentName:"tr",align:null},"The Python dictionary of observations that we discussed in ",(0,i.mdx)("a",{parentName:"td",href:"#binding-data"},"Binding Data"),".")),(0,i.mdx)("tr",{parentName:"tbody"},(0,i.mdx)("td",{parentName:"tr",align:null},(0,i.mdx)("inlineCode",{parentName:"td"},"num_samples")),(0,i.mdx)("td",{parentName:"tr",align:null},"The integer number of samples with which to approximate the posterior distributions for the values listed in ",(0,i.mdx)("inlineCode",{parentName:"td"},"queries"),".")),(0,i.mdx)("tr",{parentName:"tbody"},(0,i.mdx)("td",{parentName:"tr",align:null},(0,i.mdx)("inlineCode",{parentName:"td"},"num_adaptive_samples")),(0,i.mdx)("td",{parentName:"tr",align:null},"The integer number of samples to spend before ",(0,i.mdx)("inlineCode",{parentName:"td"},"num_samples")," on tuning the inference algorithm for the ",(0,i.mdx)("inlineCode",{parentName:"td"},"queries"),", see ",(0,i.mdx)("a",{parentName:"td",href:"/docs/adaptive_inference"},"Adaptation and Warmup"),".")),(0,i.mdx)("tr",{parentName:"tbody"},(0,i.mdx)("td",{parentName:"tr",align:null},(0,i.mdx)("inlineCode",{parentName:"td"},"num_chains")),(0,i.mdx)("td",{parentName:"tr",align:null},"The integer number of separate inference runs to use. Multiple chains can be used to verify that inference ran correctly.")))),(0,i.mdx)("p",null,"You've already seen ",(0,i.mdx)("inlineCode",{parentName:"p"},"queries")," and ",(0,i.mdx)("inlineCode",{parentName:"p"},"observations")," many times. ",(0,i.mdx)("inlineCode",{parentName:"p"},"num_adaptive_samples")," and ",(0,i.mdx)("inlineCode",{parentName:"p"},"num_samples")," are used to specify the number of iterations to respectively tune, and then run, inference. More iterations will allow inference to explore the posterior distribution more completely, resulting in more reliable posterior distributions. ",(0,i.mdx)("inlineCode",{parentName:"p"},"num_chains"),' lets you specify the number of identical runs of the entire inference algorithm to perform, called "chains". Multiple chains of inference can be used to validate that inference ran correctly and was run for enough iterations to produce reliable results, and their behavior can also help detect whether the model was well specified. We\'ll revisit chains in ',(0,i.mdx)("a",{parentName:"p",href:"/docs/inference"},"Inference Methods"),"."),(0,i.mdx)("hr",null),(0,i.mdx)("p",null,"Now that we've run inference, it's time to explore our results in the ",(0,i.mdx)("a",{parentName:"p",href:"/docs/overview/analysis/overview/analysis"},"Analysis")," section!"))}c.isMDXComponent=!0}}]);