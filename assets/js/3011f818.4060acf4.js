"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[541],{3905:function(e,a,t){t.r(a),t.d(a,{MDXContext:function(){return o},MDXProvider:function(){return c},mdx:function(){return x},useMDXComponents:function(){return d},withMDXComponents:function(){return p}});var n=t(67294);function r(e,a,t){return a in e?Object.defineProperty(e,a,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[a]=t,e}function m(){return m=Object.assign||function(e){for(var a=1;a<arguments.length;a++){var t=arguments[a];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e},m.apply(this,arguments)}function i(e,a){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);a&&(n=n.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),t.push.apply(t,n)}return t}function s(e){for(var a=1;a<arguments.length;a++){var t=null!=arguments[a]?arguments[a]:{};a%2?i(Object(t),!0).forEach((function(a){r(e,a,t[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(t,a))}))}return e}function l(e,a){if(null==e)return{};var t,n,r=function(e,a){if(null==e)return{};var t,n,r={},m=Object.keys(e);for(n=0;n<m.length;n++)t=m[n],a.indexOf(t)>=0||(r[t]=e[t]);return r}(e,a);if(Object.getOwnPropertySymbols){var m=Object.getOwnPropertySymbols(e);for(n=0;n<m.length;n++)t=m[n],a.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var o=n.createContext({}),p=function(e){return function(a){var t=d(a.components);return n.createElement(e,m({},a,{components:t}))}},d=function(e){var a=n.useContext(o),t=a;return e&&(t="function"==typeof e?e(a):s(s({},a),e)),t},c=function(e){var a=d(e.components);return n.createElement(o.Provider,{value:a},e.children)},u={inlineCode:"code",wrapper:function(e){var a=e.children;return n.createElement(n.Fragment,{},a)}},h=n.forwardRef((function(e,a){var t=e.components,r=e.mdxType,m=e.originalType,i=e.parentName,o=l(e,["components","mdxType","originalType","parentName"]),p=d(t),c=r,h=p["".concat(i,".").concat(c)]||p[c]||u[c]||m;return t?n.createElement(h,s(s({ref:a},o),{},{components:t})):n.createElement(h,s({ref:a},o))}));function x(e,a){var t=arguments,r=a&&a.mdxType;if("string"==typeof e||r){var m=t.length,i=new Array(m);i[0]=h;var s={};for(var l in a)hasOwnProperty.call(a,l)&&(s[l]=a[l]);s.originalType=e,s.mdxType="string"==typeof e?e:r,i[1]=s;for(var o=2;o<m;o++)i[o]=t[o];return n.createElement.apply(null,i)}return n.createElement.apply(null,t)}h.displayName="MDXCreateElement"},4360:function(e,a,t){t.r(a),t.d(a,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return o},toc:function(){return p},default:function(){return c}});var n=t(87462),r=t(63366),m=(t(67294),t(3905)),i=["components"],s={id:"no_u_turn_sampler",title:"No-U-Turn Sampler",sidebar_label:"No-U-Turn Sampler",slug:"/no_u_turn_sampler"},l=void 0,o={unversionedId:"framework_topics/inference/no_u_turn_sampler",id:"framework_topics/inference/no_u_turn_sampler",title:"No-U-Turn Sampler",description:"The No-U-Turn Samplers (NUTS) algorithm is an inference algorithm for differentiable random variables which uses Hamiltonian dynamics. It is an extension to the Hamiltonian Monte Carlo (HMC) inference algorithm.",source:"@site/../docs/framework_topics/inference/no_u_turn_sampler.md",sourceDirName:"framework_topics/inference",slug:"/no_u_turn_sampler",permalink:"/docs/no_u_turn_sampler",editUrl:"https://github.com/facebookresearch/beanmachine/edit/main/website/../docs/framework_topics/inference/no_u_turn_sampler.md",tags:[],version:"current",frontMatter:{id:"no_u_turn_sampler",title:"No-U-Turn Sampler",sidebar_label:"No-U-Turn Sampler",slug:"/no_u_turn_sampler"},sidebar:"someSidebar",previous:{title:"Hamiltonian Monte Carlo",permalink:"/docs/hamiltonian_monte_carlo"},next:{title:"Newtonian Monte Carlo",permalink:"/docs/newtonian_monte_carlo"}},p=[{value:"Algorithm",id:"algorithm",children:[],level:2},{value:"Usage",id:"usage",children:[],level:2}],d={toc:p};function c(e){var a=e.components,t=(0,r.Z)(e,i);return(0,m.mdx)("wrapper",(0,n.Z)({},d,t,{components:a,mdxType:"MDXLayout"}),(0,m.mdx)("p",null,"The No-U-Turn Samplers (NUTS) algorithm is an inference algorithm for differentiable random variables which uses Hamiltonian dynamics. It is an extension to the Hamiltonian Monte Carlo (HMC) inference algorithm."),(0,m.mdx)("div",{className:"admonition admonition-tip alert alert--success"},(0,m.mdx)("div",{parentName:"div",className:"admonition-heading"},(0,m.mdx)("h5",{parentName:"div"},(0,m.mdx)("span",{parentName:"h5",className:"admonition-icon"},(0,m.mdx)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,m.mdx)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,m.mdx)("div",{parentName:"div",className:"admonition-content"},(0,m.mdx)("p",{parentName:"div"},"If you haven't already read the docs on ",(0,m.mdx)("a",{parentName:"p",href:"/docs/hamiltonian_monte_carlo"},"Hamiltonian Monte Carlo"),", please read those first."))),(0,m.mdx)("h2",{id:"algorithm"},"Algorithm"),(0,m.mdx)("p",null,"The goal for NUTS is to automate the selection of an appropriate path length ",(0,m.mdx)("span",{parentName:"p",className:"math math-inline"},(0,m.mdx)("span",{parentName:"span",className:"katex"},(0,m.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,m.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.mdx)("semantics",{parentName:"math"},(0,m.mdx)("mrow",{parentName:"semantics"},(0,m.mdx)("mi",{parentName:"mrow"},"\u03bb")),(0,m.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\lambda")))),(0,m.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.mdx)("span",{parentName:"span",className:"base"},(0,m.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.69444em",verticalAlign:"0em"}}),(0,m.mdx)("span",{parentName:"span",className:"mord mathnormal"},"\u03bb")))))," for HMC inference. It extends HMC by allowing the simulation of steps backwards in time during the leapfrog step. It also uses a smart simulation algorithm that can choose a path length heuristically, so that the proposed value tends to have a low correlation with the current value."),(0,m.mdx)("p",null,"NUTS dynamically determines when the path starts looping backwards. In combination with the improvements from Adaptive HMC, this allow Bean Machine to automatically find the best step size and path length without requiring any user-tuned parameters."),(0,m.mdx)("p",null,"NUTS decides on an optimal path length by building a binary tree where each path through the binary tree represents the trajectory of a sample. Each node at depth ",(0,m.mdx)("span",{parentName:"p",className:"math math-inline"},(0,m.mdx)("span",{parentName:"span",className:"katex"},(0,m.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,m.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.mdx)("semantics",{parentName:"math"},(0,m.mdx)("mrow",{parentName:"semantics"},(0,m.mdx)("mi",{parentName:"mrow"},"j")),(0,m.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"j")))),(0,m.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.mdx)("span",{parentName:"span",className:"base"},(0,m.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.85396em",verticalAlign:"-0.19444em"}}),(0,m.mdx)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.05724em"}},"j")))))," represents simulating ",(0,m.mdx)("span",{parentName:"p",className:"math math-inline"},(0,m.mdx)("span",{parentName:"span",className:"katex"},(0,m.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,m.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.mdx)("semantics",{parentName:"math"},(0,m.mdx)("mrow",{parentName:"semantics"},(0,m.mdx)("msup",{parentName:"mrow"},(0,m.mdx)("mn",{parentName:"msup"},"2"),(0,m.mdx)("mi",{parentName:"msup"},"j"))),(0,m.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"2^j")))),(0,m.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.mdx)("span",{parentName:"span",className:"base"},(0,m.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.824664em",verticalAlign:"0em"}}),(0,m.mdx)("span",{parentName:"span",className:"mord"},(0,m.mdx)("span",{parentName:"span",className:"mord"},"2"),(0,m.mdx)("span",{parentName:"span",className:"msupsub"},(0,m.mdx)("span",{parentName:"span",className:"vlist-t"},(0,m.mdx)("span",{parentName:"span",className:"vlist-r"},(0,m.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.824664em"}},(0,m.mdx)("span",{parentName:"span",style:{top:"-3.063em",marginRight:"0.05em"}},(0,m.mdx)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.mdx)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.mdx)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.05724em"}},"j"))))))))))))," steps forwards or backwards. This binary tree is adaptively grown until either hitting a pre-specified max depth size, or until the algorithm starts proposing samples with too low probabilities due to discretization errors."),(0,m.mdx)("p",null,"The full NUTS algorithm description is quite involved. We recommend you check out ",(0,m.mdx)("a",{parentName:"p",href:"https://arxiv.org/pdf/1111.4246.pdf"},"Hoffman & Gelman, 2011")," to learn more."),(0,m.mdx)("h2",{id:"usage"},"Usage"),(0,m.mdx)("p",null,"Bean Machine provides a single-site version of NUTS that only updates one variable at a time:"),(0,m.mdx)("pre",null,(0,m.mdx)("code",{parentName:"pre",className:"language-py"},"bm.SingleSiteNoUTurnSampler(\n    use_dense_mass_matrix=True,\n).infer(\n    queries,\n    observations,\n    num_samples,\n    num_chains,\n    num_adaptive_samples=1000,\n)\n")),(0,m.mdx)("div",{className:"admonition admonition-caution alert alert--warning"},(0,m.mdx)("div",{parentName:"div",className:"admonition-heading"},(0,m.mdx)("h5",{parentName:"div"},(0,m.mdx)("span",{parentName:"h5",className:"admonition-icon"},(0,m.mdx)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"16",height:"16",viewBox:"0 0 16 16"},(0,m.mdx)("path",{parentName:"svg",fillRule:"evenodd",d:"M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"}))),"caution")),(0,m.mdx)("div",{parentName:"div",className:"admonition-content"},(0,m.mdx)("p",{parentName:"div"},(0,m.mdx)("strong",{parentName:"p"},"Make sure to set ",(0,m.mdx)("inlineCode",{parentName:"strong"},"num_adaptive_samples")," when using adaptive HMC!")," If you forget to set ",(0,m.mdx)("inlineCode",{parentName:"p"},"num_adaptive_samples"),", no adaptation will occur."))),(0,m.mdx)("p",null,"Bean Machine also provides a multi-site version of NUTS that updates all variables in your model at the same time. This is only appropriate for models that are comprised of only continuous random variables."),(0,m.mdx)("pre",null,(0,m.mdx)("code",{parentName:"pre",className:"language-py"},"bm.GlobalNoUTurnSampler(\n    max_tree_depth=10,\n    max_delta_energy=1000.0,\n    initial_step_size=1.0,\n    adapt_step_size=True,\n    adapt_mass_matrix=True,\n    multinomial_sampling=True,\n    target_accept_prob=0.8,\n).infer(\n    queries,\n    observations,\n    num_samples,\n    num_chains,\n    num_adaptive_samples=1000,\n)\n")),(0,m.mdx)("p",null,"The ",(0,m.mdx)("inlineCode",{parentName:"p"},"GlobalNoUTurnSampler")," has all the acceptance step size, covariance matrix, and acceptance probability tuning arguments of ",(0,m.mdx)("inlineCode",{parentName:"p"},"GlobalHamiltonianMonteCarlo")," as well as a few more parameters related to tuning the path length. While there are many optional parameters for this inference method, in practice, the parameters you are most likely to modify are ",(0,m.mdx)("inlineCode",{parentName:"p"},"target_accept_prob")," and ",(0,m.mdx)("inlineCode",{parentName:"p"},"max_tree_depth"),". When dealing with posteriors where the probability density has a more complicated shape, we benefit from taking smaller steps. Setting ",(0,m.mdx)("inlineCode",{parentName:"p"},"target_accept_prob")," to a higher value like ",(0,m.mdx)("inlineCode",{parentName:"p"},"0.9")," will need to a more careful exploration of the space using smaller step sizes while still benefiting from some tuning of that step size. Since we will be taking smaller steps we need to compensate by having a larger path length. This is accomplished by increasing ",(0,m.mdx)("inlineCode",{parentName:"p"},"max_tree_depth"),". Otherwise, using the defaults provided is highly recommended."),(0,m.mdx)("p",null,"A more complete explanation of parameters to ",(0,m.mdx)("inlineCode",{parentName:"p"},"GlobalNoUTurnSampler")," are provided below:"),(0,m.mdx)("table",null,(0,m.mdx)("thead",{parentName:"table"},(0,m.mdx)("tr",{parentName:"thead"},(0,m.mdx)("th",{parentName:"tr",align:null},"Name"),(0,m.mdx)("th",{parentName:"tr",align:null},"Usage"))),(0,m.mdx)("tbody",{parentName:"table"},(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"max_tree_depth")),(0,m.mdx)("td",{parentName:"tr",align:null},"The maximum depth of the binary tree used to simulate leapfrog steps forwards and backwards in time.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"max_delta_energy")),(0,m.mdx)("td",{parentName:"tr",align:null},"This is the lowest probability moves that NUTS will consider. Once most new samples have a lower probability, NUTS will stop its leapfrog steps. This should be interpreted as a negative log probability and is designed to be fairly conservative.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"initial_step_size")),(0,m.mdx)("td",{parentName:"tr",align:null},"The initial step size ",(0,m.mdx)("span",{parentName:"td",className:"math math-inline"},(0,m.mdx)("span",{parentName:"span",className:"katex"},(0,m.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,m.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.mdx)("semantics",{parentName:"math"},(0,m.mdx)("mrow",{parentName:"semantics"},(0,m.mdx)("mi",{parentName:"mrow"},"\u03f5")),(0,m.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\epsilon")))),(0,m.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.mdx)("span",{parentName:"span",className:"base"},(0,m.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.43056em",verticalAlign:"0em"}}),(0,m.mdx)("span",{parentName:"span",className:"mord mathnormal"},"\u03f5")))))," used in adaptive HMC. This value is simply the step size if tuning is disabled.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"multinomial_sampling")),(0,m.mdx)("td",{parentName:"tr",align:null},"Lets us decide between a faster multinomial sampler for the trajectory or the slice sampler described in the ",(0,m.mdx)("a",{parentName:"td",href:"https://arxiv.org/pdf/1111.4246.pdf"},"original paper"),". The option is useful for fairly comparing against other NUTS implementations.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"target_accept_prob")),(0,m.mdx)("td",{parentName:"tr",align:null},"Indicates the acceptance probability which should be targeted by the step size tuning algorithm. While the optimal value is 65.1%, higher values have been show to be more robust leading to a default of 0.8.")))),(0,m.mdx)("p",null,"The parameters to ",(0,m.mdx)("inlineCode",{parentName:"p"},"infer")," are described below:"),(0,m.mdx)("table",null,(0,m.mdx)("thead",{parentName:"table"},(0,m.mdx)("tr",{parentName:"thead"},(0,m.mdx)("th",{parentName:"tr",align:null},"Name"),(0,m.mdx)("th",{parentName:"tr",align:null},"Usage"))),(0,m.mdx)("tbody",{parentName:"table"},(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"queries")),(0,m.mdx)("td",{parentName:"tr",align:null},"A ",(0,m.mdx)("inlineCode",{parentName:"td"},"List")," of ",(0,m.mdx)("inlineCode",{parentName:"td"},"@bm.random_variable")," targets to fit posterior distributions for.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"observations")),(0,m.mdx)("td",{parentName:"tr",align:null},"The ",(0,m.mdx)("inlineCode",{parentName:"td"},"Dict")," of observations. Each key is a random variable, and its value is the observed value for that random variable.")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"num_samples")),(0,m.mdx)("td",{parentName:"tr",align:null},"Number of samples to build up distributions for the values listed in ",(0,m.mdx)("inlineCode",{parentName:"td"},"queries"),".")),(0,m.mdx)("tr",{parentName:"tbody"},(0,m.mdx)("td",{parentName:"tr",align:null},(0,m.mdx)("inlineCode",{parentName:"td"},"num_chains")),(0,m.mdx)("td",{parentName:"tr",align:null},"Number of separate inference runs to use. Multiple chains can be used by diagnostics to verify inference ran correctly.")))),(0,m.mdx)("hr",null),(0,m.mdx)("p",null,'Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." ',(0,m.mdx)("em",{parentName:"p"},"J. Mach. Learn. Res.")," 15.1 (2014): 1593-1623."))}c.isMDXComponent=!0}}]);