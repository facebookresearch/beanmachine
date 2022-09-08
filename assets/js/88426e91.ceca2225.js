"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[3176],{3905:function(e,a,t){t.r(a),t.d(a,{MDXContext:function(){return o},MDXProvider:function(){return c},mdx:function(){return x},useMDXComponents:function(){return d},withMDXComponents:function(){return p}});var n=t(67294);function r(e,a,t){return a in e?Object.defineProperty(e,a,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[a]=t,e}function m(){return m=Object.assign||function(e){for(var a=1;a<arguments.length;a++){var t=arguments[a];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e},m.apply(this,arguments)}function i(e,a){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);a&&(n=n.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),t.push.apply(t,n)}return t}function s(e){for(var a=1;a<arguments.length;a++){var t=null!=arguments[a]?arguments[a]:{};a%2?i(Object(t),!0).forEach((function(a){r(e,a,t[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(t,a))}))}return e}function l(e,a){if(null==e)return{};var t,n,r=function(e,a){if(null==e)return{};var t,n,r={},m=Object.keys(e);for(n=0;n<m.length;n++)t=m[n],a.indexOf(t)>=0||(r[t]=e[t]);return r}(e,a);if(Object.getOwnPropertySymbols){var m=Object.getOwnPropertySymbols(e);for(n=0;n<m.length;n++)t=m[n],a.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var o=n.createContext({}),p=function(e){return function(a){var t=d(a.components);return n.createElement(e,m({},a,{components:t}))}},d=function(e){var a=n.useContext(o),t=a;return e&&(t="function"==typeof e?e(a):s(s({},a),e)),t},c=function(e){var a=d(e.components);return n.createElement(o.Provider,{value:a},e.children)},u={inlineCode:"code",wrapper:function(e){var a=e.children;return n.createElement(n.Fragment,{},a)}},h=n.forwardRef((function(e,a){var t=e.components,r=e.mdxType,m=e.originalType,i=e.parentName,o=l(e,["components","mdxType","originalType","parentName"]),p=d(t),c=r,h=p["".concat(i,".").concat(c)]||p[c]||u[c]||m;return t?n.createElement(h,s(s({ref:a},o),{},{components:t})):n.createElement(h,s({ref:a},o))}));function x(e,a){var t=arguments,r=a&&a.mdxType;if("string"==typeof e||r){var m=t.length,i=new Array(m);i[0]=h;var s={};for(var l in a)hasOwnProperty.call(a,l)&&(s[l]=a[l]);s.originalType=e,s.mdxType="string"==typeof e?e:r,i[1]=s;for(var o=2;o<m;o++)i[o]=t[o];return n.createElement.apply(null,i)}return n.createElement.apply(null,t)}h.displayName="MDXCreateElement"},44307:function(e,a,t){t.r(a),t.d(a,{assets:function(){return p},contentTitle:function(){return l},default:function(){return u},frontMatter:function(){return s},metadata:function(){return o},toc:function(){return d}});var n=t(83117),r=t(80102),m=(t(67294),t(3905)),i=["components"],s={id:"no_u_turn_sampler",title:"No-U-Turn Sampler",sidebar_label:"No-U-Turn Sampler",slug:"/no_u_turn_sampler"},l=void 0,o={unversionedId:"framework_topics/mcmc_inference/no_u_turn_sampler",id:"framework_topics/mcmc_inference/no_u_turn_sampler",title:"No-U-Turn Sampler",description:"The No-U-Turn Sampler (NUTS) (Hoffman and Gelman, 2014) algorithm is an inference algorithm for differentiable random variables which uses Hamiltonian dynamics. It is an extension to the Hamiltonian Monte Carlo (HMC) inference algorithm.",source:"@site/../docs/framework_topics/mcmc_inference/no_u_turn_sampler.md",sourceDirName:"framework_topics/mcmc_inference",slug:"/no_u_turn_sampler",permalink:"/docs/no_u_turn_sampler",draft:!1,editUrl:"https://github.com/facebookresearch/beanmachine/edit/main/website/../docs/framework_topics/mcmc_inference/no_u_turn_sampler.md",tags:[],version:"current",frontMatter:{id:"no_u_turn_sampler",title:"No-U-Turn Sampler",sidebar_label:"No-U-Turn Sampler",slug:"/no_u_turn_sampler"},sidebar:"someSidebar",previous:{title:"Hamiltonian Monte Carlo",permalink:"/docs/hamiltonian_monte_carlo"},next:{title:"Newtonian Monte Carlo",permalink:"/docs/newtonian_monte_carlo"}},p={},d=[{value:"Algorithm",id:"algorithm",level:2},{value:"Usage",id:"usage",level:2}],c={toc:d};function u(e){var a=e.components,t=(0,r.Z)(e,i);return(0,m.mdx)("wrapper",(0,n.Z)({},c,t,{components:a,mdxType:"MDXLayout"}),(0,m.mdx)("p",null,"The No-U-Turn Sampler (NUTS) (Hoffman and Gelman, 2014) algorithm is an inference algorithm for differentiable random variables which uses Hamiltonian dynamics. It is an extension to the Hamiltonian Monte Carlo (HMC) inference algorithm."),(0,m.mdx)("admonition",{type:"tip"},(0,m.mdx)("p",{parentName:"admonition"},"If you haven't already read the docs on ",(0,m.mdx)("a",{parentName:"p",href:"/docs/hamiltonian_monte_carlo"},"Hamiltonian Monte Carlo"),", please read those first.")),(0,m.mdx)("h2",{id:"algorithm"},"Algorithm"),(0,m.mdx)("p",null,"The goal for NUTS is to automate the selection of an appropriate path length ",(0,m.mdx)("span",{parentName:"p",className:"math math-inline"},(0,m.mdx)("span",{parentName:"span",className:"katex"},(0,m.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,m.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.mdx)("semantics",{parentName:"math"},(0,m.mdx)("mrow",{parentName:"semantics"},(0,m.mdx)("mi",{parentName:"mrow"},"\u03bb")),(0,m.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\lambda")))),(0,m.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.mdx)("span",{parentName:"span",className:"base"},(0,m.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.69444em",verticalAlign:"0em"}}),(0,m.mdx)("span",{parentName:"span",className:"mord mathnormal"},"\u03bb")))))," for HMC inference. It extends HMC by allowing the simulation of steps backwards in time during the leapfrog step. It also uses a smart simulation algorithm that can choose a path length heuristically, so that the proposed value tends to have a low correlation with the current value."),(0,m.mdx)("p",null,"NUTS dynamically determines when the path starts looping backwards. In combination with the improvements from Adaptive HMC, this allow Bean Machine to automatically find the best step size and path length without requiring any user-tuned parameters."),(0,m.mdx)("p",null,"NUTS decides on an optimal path length by building a binary tree where each path through the binary tree represents the trajectory of a sample. Each node at depth ",(0,m.mdx)("span",{parentName:"p",className:"math math-inline"},(0,m.mdx)("span",{parentName:"span",className:"katex"},(0,m.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,m.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.mdx)("semantics",{parentName:"math"},(0,m.mdx)("mrow",{parentName:"semantics"},(0,m.mdx)("mi",{parentName:"mrow"},"j")),(0,m.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"j")))),(0,m.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.mdx)("span",{parentName:"span",className:"base"},(0,m.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.85396em",verticalAlign:"-0.19444em"}}),(0,m.mdx)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.05724em"}},"j")))))," represents simulating ",(0,m.mdx)("span",{parentName:"p",className:"math math-inline"},(0,m.mdx)("span",{parentName:"span",className:"katex"},(0,m.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,m.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.mdx)("semantics",{parentName:"math"},(0,m.mdx)("mrow",{parentName:"semantics"},(0,m.mdx)("msup",{parentName:"mrow"},(0,m.mdx)("mn",{parentName:"msup"},"2"),(0,m.mdx)("mi",{parentName:"msup"},"j"))),(0,m.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"2^j")))),(0,m.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.mdx)("span",{parentName:"span",className:"base"},(0,m.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.824664em",verticalAlign:"0em"}}),(0,m.mdx)("span",{parentName:"span",className:"mord"},(0,m.mdx)("span",{parentName:"span",className:"mord"},"2"),(0,m.mdx)("span",{parentName:"span",className:"msupsub"},(0,m.mdx)("span",{parentName:"span",className:"vlist-t"},(0,m.mdx)("span",{parentName:"span",className:"vlist-r"},(0,m.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.824664em"}},(0,m.mdx)("span",{parentName:"span",style:{top:"-3.063em",marginRight:"0.05em"}},(0,m.mdx)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.mdx)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.mdx)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.05724em"}},"j"))))))))))))," steps forwards or backwards. This binary tree is adaptively grown until either hitting a pre-specified max depth size, or until the algorithm starts proposing samples with too low probabilities due to discretization errors."),(0,m.mdx)("p",null,"The full NUTS algorithm description is quite involved. We recommend you check out ",(0,m.mdx)("a",{parentName:"p",href:"https://arxiv.org/pdf/1111.4246.pdf"},"Hoffman & Gelman, 2011")," to learn more."),(0,m.mdx)("admonition",{type:"caution"},(0,m.mdx)("p",{parentName:"admonition"},"As with HMC, NUTS operates on continuous latent variables only. For discrete variables, use ",(0,m.mdx)("a",{parentName:"p",href:"/docs/compositional_inference"},(0,m.mdx)("inlineCode",{parentName:"a"},"CompositionalInference"))," or marginalize them out as in the ",(0,m.mdx)("a",{parentName:"p",href:"../overview/tutorials/Zero_inflated_count_data/ZeroInflatedCountData"},"Zero inflated count data tutorial"),".")),(0,m.mdx)("h2",{id:"usage"},"Usage"),(0,m.mdx)("p",null,"Bean Machine provides a single-site version of NUTS that only updates one variable at a time\nas well as a multi-site version of NUTS that updates all the variables in your model jointly at each step. Both follow the same API:"),(0,m.mdx)("pre",null,(0,m.mdx)("code",{parentName:"pre",className:"language-py"},"bm.SingleSiteNoUTurnSampler(\n    max_tree_depth=10,\n    max_delta_energy=1000.0,\n    initial_step_size=1.0,\n    adapt_step_size=True,\n    adapt_mass_matrix=True,\n    multinomial_sampling=True,\n    target_accept_prob=0.8,\n).infer(\n    queries,\n    observations,\n    num_samples,\n    num_chains,\n    num_adaptive_samples=1000,\n)\n\nbm.GlobalNoUTurnSampler(\n    max_tree_depth=10,\n    max_delta_energy=1000.0,\n    initial_step_size=1.0,\n    adapt_step_size=True,\n    adapt_mass_matrix=True,\n    multinomial_sampling=True,\n    target_accept_prob=0.8,\n).infer(\n    queries,\n    observations,\n    num_samples,\n    num_chains,\n    num_adaptive_samples=1000,\n)\n")),(0,m.mdx)("admonition",{type:"caution"},(0,m.mdx)("p",{parentName:"admonition"},"Functorch's ",(0,m.mdx)("a",{parentName:"p",href:"https://pytorch.org/functorch/stable/aot_autograd.html"},"ahead of time (AOT) autograd compiler")," is used\nby default. If working with a non-static model or unexpected errors are encountered, you may need to manually\ndisable the ",(0,m.mdx)("inlineCode",{parentName:"p"},"nnc_compile")," flag.")),(0,m.mdx)("p",null,"The ",(0,m.mdx)("inlineCode",{parentName:"p"},"GlobalNoUTurnSampler")," has all the acceptance step size, covariance matrix, and acceptance probability tuning arguments of ",(0,m.mdx)("inlineCode",{parentName:"p"},"GlobalHamiltonianMonteCarlo")," as well as a few more parameters related to tuning the path length. While there are many optional parameters for this inference method, in practice, the parameters you are most likely to modify are ",(0,m.mdx)("inlineCode",{parentName:"p"},"target_accept_prob")," and ",(0,m.mdx)("inlineCode",{parentName:"p"},"max_tree_depth"),". When dealing with posteriors where the probability density has a more complicated shape, we benefit from taking smaller steps. Setting ",(0,m.mdx)("inlineCode",{parentName:"p"},"target_accept_prob")," to a higher value like ",(0,m.mdx)("inlineCode",{parentName:"p"},"0.9")," will lead to a more careful exploration of the space using smaller step sizes while still benefiting from some tuning of that step size. Since we will be taking smaller steps, we need to compensate by having a larger path length. This is accomplished by increasing ",(0,m.mdx)("inlineCode",{parentName:"p"},"max_tree_depth"),". Otherwise, using the defaults provided is highly recommended."),(0,m.mdx)("p",null,"A more complete explanation of parameters to ",(0,m.mdx)("inlineCode",{parentName:"p"},"GlobalNoUTurnSampler")," are provided below and in the ",(0,m.mdx)("a",{parentName:"p",href:"https://beanmachine.org/api/beanmachine.ppl.html?highlight=nouturnsampler#beanmachine.ppl.GlobalNoUTurnSampler"},"docs"),":"),(0,m.mdx)("table",null,(0,m.mdx)("thead",{parentName:"table"},(0,m.mdx)("tr",{parentName:"thead"},(0,m.mdx)("th",{parentName:"tr",align:null},"Name"),(0,m.mdx)("th",{parentName:"tr",align:null},"Usage"))),(0,m.mdx)("tbody",{parentName:"table"},(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"max_tree_depth")),(0,m.mdx)("td",{parentName:"tr",align:null},"The maximum depth of the binary tree used to simulate leapfrog steps forwards and backwards in time.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"max_delta_energy")),(0,m.mdx)("td",{parentName:"tr",align:null},"This is the lowest probability moves that NUTS will consider. Once most new samples have a lower probability, NUTS will stop its leapfrog steps. This should be interpreted as a negative log probability and is designed to be fairly conservative.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"initial_step_size")),(0,m.mdx)("td",{parentName:"tr",align:null},"The initial step size ",(0,m.mdx)("span",{parentName:"td",className:"math math-inline"},(0,m.mdx)("span",{parentName:"span",className:"katex"},(0,m.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,m.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.mdx)("semantics",{parentName:"math"},(0,m.mdx)("mrow",{parentName:"semantics"},(0,m.mdx)("mi",{parentName:"mrow"},"\u03f5")),(0,m.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\epsilon")))),(0,m.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.mdx)("span",{parentName:"span",className:"base"},(0,m.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.43056em",verticalAlign:"0em"}}),(0,m.mdx)("span",{parentName:"span",className:"mord mathnormal"},"\u03f5")))))," used in adaptive HMC. This value is simply the step size if tuning is disabled.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"multinomial_sampling")),(0,m.mdx)("td",{parentName:"tr",align:null},"Lets us decide between a faster multinomial sampler for the trajectory or the slice sampler described in the ",(0,m.mdx)("a",{parentName:"td",href:"https://arxiv.org/pdf/1111.4246.pdf"},"original paper"),". The option is useful for fairly comparing against other NUTS implementations.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"target_accept_prob")),(0,m.mdx)("td",{parentName:"tr",align:null},"Indicates the acceptance probability which should be targeted by the step size tuning algorithm. While the optimal value is 65.1%, higher values have been show to be more robust leading to a default of 0.8.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"nnc_compile")),(0,m.mdx)("td",{parentName:"tr",align:null},"NNC (neural network compiler) is a Pytorch JIT compiler that that transforms Pytorch programs to LLVM-compiled binaries. The model support is currently limited, so if your model fails, consider filing an issue and turning this flag off.")))),(0,m.mdx)("p",null,"The parameters to ",(0,m.mdx)("inlineCode",{parentName:"p"},"infer")," are described below:"),(0,m.mdx)("table",null,(0,m.mdx)("thead",{parentName:"table"},(0,m.mdx)("tr",{parentName:"thead"},(0,m.mdx)("th",{parentName:"tr",align:null},"Name"),(0,m.mdx)("th",{parentName:"tr",align:null},"Usage"))),(0,m.mdx)("tbody",{parentName:"table"},(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"queries")),(0,m.mdx)("td",{parentName:"tr",align:null},"A ",(0,m.mdx)("inlineCode",{parentName:"td"},"List")," of ",(0,m.mdx)("inlineCode",{parentName:"td"},"@bm.random_variable")," targets to fit posterior distributions for.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"observations")),(0,m.mdx)("td",{parentName:"tr",align:null},"The ",(0,m.mdx)("inlineCode",{parentName:"td"},"Dict")," of observations. Each key is a random variable, and its value is the observed value for that random variable.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"num_samples")),(0,m.mdx)("td",{parentName:"tr",align:null},"Number of samples to build up distributions for the values listed in ",(0,m.mdx)("inlineCode",{parentName:"td"},"queries"),".")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"num_chains")),(0,m.mdx)("td",{parentName:"tr",align:null},"Number of separate inference runs to use. Multiple chains can be used by diagnostics to verify inference ran correctly.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"num_adaptive_samples")),(0,m.mdx)("td",{parentName:"tr",align:null},"Number of warmup samples to adapt the parameters.")))),(0,m.mdx)("hr",null),(0,m.mdx)("p",null,'Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." ',(0,m.mdx)("em",{parentName:"p"},"J. Mach. Learn. Res.")," 15.1 (2014): 1593-1623."))}u.isMDXComponent=!0}}]);