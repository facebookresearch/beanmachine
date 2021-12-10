"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[2709],{3905:function(a,e,n){n.r(e),n.d(e,{MDXContext:function(){return o},MDXProvider:function(){return c},mdx:function(){return h},useMDXComponents:function(){return d},withMDXComponents:function(){return l}});var t=n(67294);function m(a,e,n){return e in a?Object.defineProperty(a,e,{value:n,enumerable:!0,configurable:!0,writable:!0}):a[e]=n,a}function s(){return s=Object.assign||function(a){for(var e=1;e<arguments.length;e++){var n=arguments[e];for(var t in n)Object.prototype.hasOwnProperty.call(n,t)&&(a[t]=n[t])}return a},s.apply(this,arguments)}function r(a,e){var n=Object.keys(a);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(a);e&&(t=t.filter((function(e){return Object.getOwnPropertyDescriptor(a,e).enumerable}))),n.push.apply(n,t)}return n}function i(a){for(var e=1;e<arguments.length;e++){var n=null!=arguments[e]?arguments[e]:{};e%2?r(Object(n),!0).forEach((function(e){m(a,e,n[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(a,Object.getOwnPropertyDescriptors(n)):r(Object(n)).forEach((function(e){Object.defineProperty(a,e,Object.getOwnPropertyDescriptor(n,e))}))}return a}function p(a,e){if(null==a)return{};var n,t,m=function(a,e){if(null==a)return{};var n,t,m={},s=Object.keys(a);for(t=0;t<s.length;t++)n=s[t],e.indexOf(n)>=0||(m[n]=a[n]);return m}(a,e);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(a);for(t=0;t<s.length;t++)n=s[t],e.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(a,n)&&(m[n]=a[n])}return m}var o=t.createContext({}),l=function(a){return function(e){var n=d(e.components);return t.createElement(a,s({},e,{components:n}))}},d=function(a){var e=t.useContext(o),n=e;return a&&(n="function"==typeof a?a(e):i(i({},e),a)),n},c=function(a){var e=d(a.components);return t.createElement(o.Provider,{value:e},a.children)},N={inlineCode:"code",wrapper:function(a){var e=a.children;return t.createElement(t.Fragment,{},e)}},x=t.forwardRef((function(a,e){var n=a.components,m=a.mdxType,s=a.originalType,r=a.parentName,o=p(a,["components","mdxType","originalType","parentName"]),l=d(n),c=m,x=l["".concat(r,".").concat(c)]||l[c]||N[c]||s;return n?t.createElement(x,i(i({ref:e},o),{},{components:n})):t.createElement(x,i({ref:e},o))}));function h(a,e){var n=arguments,m=e&&e.mdxType;if("string"==typeof a||m){var s=n.length,r=new Array(s);r[0]=x;var i={};for(var p in e)hasOwnProperty.call(e,p)&&(i[p]=e[p]);i.originalType=a,i.mdxType="string"==typeof a?a:m,r[1]=i;for(var o=2;o<s;o++)r[o]=n[o];return t.createElement.apply(null,r)}return t.createElement.apply(null,n)}x.displayName="MDXCreateElement"},19431:function(a,e,n){n.r(e),n.d(e,{frontMatter:function(){return i},contentTitle:function(){return p},metadata:function(){return o},toc:function(){return l},default:function(){return c}});var t=n(87462),m=n(63366),s=(n(67294),n(3905)),r=["components"],i={id:"modeling",title:"Modeling",sidebar_label:"Modeling"},p=void 0,o={unversionedId:"overview/modeling/modeling",id:"overview/modeling/modeling",title:"Modeling",description:"Declarative Style",source:"@site/../docs/overview/modeling/modeling.md",sourceDirName:"overview/modeling",slug:"/overview/modeling/",permalink:"/docs/overview/modeling/",editUrl:"https://github.com/facebookresearch/beanmachine/edit/main/website/../docs/overview/modeling/modeling.md",tags:[],version:"current",frontMatter:{id:"modeling",title:"Modeling",sidebar_label:"Modeling"},sidebar:"someSidebar",previous:{title:"Quick Start",permalink:"/docs/quickstart"},next:{title:"Inference",permalink:"/docs/overview/inference/"}},l=[{value:'<a name="declarative_style"></a>Declarative Style',id:"declarative-style",children:[],level:2},{value:"Random Variable Functions",id:"random-variable-functions",children:[],level:2},{value:'<a name="calling_inside"></a>Calling a Random Variable from Another Random Variable Function',id:"calling-a-random-variable-from-another-random-variable-function",children:[],level:2},{value:'<a name="calling_outside"></a>Calling a Random Variable from an Ordinary Function',id:"calling-a-random-variable-from-an-ordinary-function",children:[],level:2},{value:'<a name="random_variable_families"></a>Defining Random Variable Families',id:"defining-random-variable-families",children:[],level:2},{value:"Transforming Random Variables",id:"transforming-random-variables",children:[],level:2}],d={toc:l};function c(a){var e=a.components,n=(0,m.Z)(a,r);return(0,s.mdx)("wrapper",(0,t.Z)({},d,n,{components:e,mdxType:"MDXLayout"}),(0,s.mdx)("h2",{id:"declarative-style"},(0,s.mdx)("a",{name:"declarative_style"}),"Declarative Style"),(0,s.mdx)("p",null,"Bean Machine allows you to express models declaratively, in a way that closely follows the notation that statisticians use in their everyday work. Consider our example from the ",(0,s.mdx)("a",{parentName:"p",href:"/docs/quickstart"},"Quick Start"),". We could express this mathematically as:"),(0,s.mdx)("ul",null,(0,s.mdx)("li",{parentName:"ul"},(0,s.mdx)("span",{parentName:"li",className:"math math-inline"},(0,s.mdx)("span",{parentName:"span",className:"katex"},(0,s.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,s.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,s.mdx)("semantics",{parentName:"math"},(0,s.mdx)("mrow",{parentName:"semantics"},(0,s.mdx)("msub",{parentName:"mrow"},(0,s.mdx)("mi",{parentName:"msub"},"n"),(0,s.mdx)("mtext",{parentName:"msub"},"infected"))),(0,s.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"n_\\text{infected}")))),(0,s.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.58056em",verticalAlign:"-0.15em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal"},"n"),(0,s.mdx)("span",{parentName:"span",className:"msupsub"},(0,s.mdx)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.33610799999999996em"}},(0,s.mdx)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,s.mdx)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,s.mdx)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord text mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord mtight"},"infected"))))),(0,s.mdx)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,s.mdx)("span",{parentName:"span"})))))))))),": known constant"),(0,s.mdx)("li",{parentName:"ul"},(0,s.mdx)("span",{parentName:"li",className:"math math-inline"},(0,s.mdx)("span",{parentName:"span",className:"katex"},(0,s.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,s.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,s.mdx)("semantics",{parentName:"math"},(0,s.mdx)("mrow",{parentName:"semantics"},(0,s.mdx)("mtext",{parentName:"mrow",mathvariant:"monospace"},"reproduction_rate"),(0,s.mdx)("mo",{parentName:"mrow"},"\u223c"),(0,s.mdx)("mtext",{parentName:"mrow"},"Exponential"),(0,s.mdx)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,s.mdx)("mn",{parentName:"mrow"},"10.0"),(0,s.mdx)("mo",{parentName:"mrow",stretchy:"false"},")")),(0,s.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\texttt{reproduction\\_rate} \\sim \\text{Exponential}(10.0)")))),(0,s.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.83333em",verticalAlign:"-0.22222em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord text"},(0,s.mdx)("span",{parentName:"span",className:"mord texttt"},"reproduction_rate")),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2777777777777778em"}}),(0,s.mdx)("span",{parentName:"span",className:"mrel"},"\u223c"),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2777777777777778em"}})),(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord text"},(0,s.mdx)("span",{parentName:"span",className:"mord"},"Exponential")),(0,s.mdx)("span",{parentName:"span",className:"mopen"},"("),(0,s.mdx)("span",{parentName:"span",className:"mord"},"1"),(0,s.mdx)("span",{parentName:"span",className:"mord"},"0"),(0,s.mdx)("span",{parentName:"span",className:"mord"},"."),(0,s.mdx)("span",{parentName:"span",className:"mord"},"0"),(0,s.mdx)("span",{parentName:"span",className:"mclose"},")")))))),(0,s.mdx)("li",{parentName:"ul"},(0,s.mdx)("span",{parentName:"li",className:"math math-inline"},(0,s.mdx)("span",{parentName:"span",className:"katex"},(0,s.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,s.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,s.mdx)("semantics",{parentName:"math"},(0,s.mdx)("mrow",{parentName:"semantics"},(0,s.mdx)("msub",{parentName:"mrow"},(0,s.mdx)("mi",{parentName:"msub"},"n"),(0,s.mdx)("mtext",{parentName:"msub"},"new")),(0,s.mdx)("mo",{parentName:"mrow"},"\u223c"),(0,s.mdx)("mtext",{parentName:"mrow"},"Poisson"),(0,s.mdx)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,s.mdx)("mtext",{parentName:"mrow",mathvariant:"monospace"},"reproduction_rate"),(0,s.mdx)("mo",{parentName:"mrow"},"\u22c5"),(0,s.mdx)("msub",{parentName:"mrow"},(0,s.mdx)("mi",{parentName:"msub"},"n"),(0,s.mdx)("mtext",{parentName:"msub"},"infected")),(0,s.mdx)("mo",{parentName:"mrow",stretchy:"false"},")")),(0,s.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"n_\\text{new} \\sim \\text{Poisson}(\\texttt{reproduction\\_rate} \\cdot n_\\text{infected})")))),(0,s.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.58056em",verticalAlign:"-0.15em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal"},"n"),(0,s.mdx)("span",{parentName:"span",className:"msupsub"},(0,s.mdx)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.151392em"}},(0,s.mdx)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,s.mdx)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,s.mdx)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord text mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord mtight"},"new"))))),(0,s.mdx)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,s.mdx)("span",{parentName:"span"})))))),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2777777777777778em"}}),(0,s.mdx)("span",{parentName:"span",className:"mrel"},"\u223c"),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2777777777777778em"}})),(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord text"},(0,s.mdx)("span",{parentName:"span",className:"mord"},"Poisson")),(0,s.mdx)("span",{parentName:"span",className:"mopen"},"("),(0,s.mdx)("span",{parentName:"span",className:"mord text"},(0,s.mdx)("span",{parentName:"span",className:"mord texttt"},"reproduction_rate")),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222222222222222em"}}),(0,s.mdx)("span",{parentName:"span",className:"mbin"},"\u22c5"),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222222222222222em"}})),(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal"},"n"),(0,s.mdx)("span",{parentName:"span",className:"msupsub"},(0,s.mdx)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.33610799999999996em"}},(0,s.mdx)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,s.mdx)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,s.mdx)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord text mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord mtight"},"infected"))))),(0,s.mdx)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,s.mdx)("span",{parentName:"span"})))))),(0,s.mdx)("span",{parentName:"span",className:"mclose"},")"))))))),(0,s.mdx)("p",null,"Let's take a look at the model again:"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-py"},"num_infected = 1087980\n\n@bm.random_variable\ndef reproduction_rate():\n    return dist.Exponential(rate=10.0)\n\n@bm.random_variable\ndef num_new_cases():\n    return dist.Poisson(reproduction_rate() *  num_infected)\n")),(0,s.mdx)("p",null,"You can see how the Python code maps almost one-to-one to the mathematical definition. When building models in Bean Machine's declarative syntax, we encourage you to first think of the model mathematically, and then to evolve the code to fit to that definition."),(0,s.mdx)("p",null,"Importantly, note that there is no formal class delineating your model. This means you're maximally free to build models that feel organic with the rest of your codebase and compose seamlessly with models found elsewhere in your codebase. Of course, you're also free to consolidate related modeling functionality within a class, which can help keep your model appropriately scoped!"),(0,s.mdx)("h2",{id:"random-variable-functions"},"Random Variable Functions"),(0,s.mdx)("p",null,"Python functions annotated with ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.random_variable"),', or "random variable functions" for short, are the building blocks of models in Bean Machine. This decorator denotes functions which should be treated by the framework as random variables to learn about.'),(0,s.mdx)("p",null,"A random variable function must return a ",(0,s.mdx)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/distributions.html?highlight=distribution#module-torch.distributions"},"PyTorch distribution")," representing the probability distribution for that random variable, conditional on sample values for any other random variable functions that it depends on. For the most part, random variable functions can contain arbitrary Python code to model your problem! However, please do not depend on mutable external state (such as Python's ",(0,s.mdx)("inlineCode",{parentName:"p"},"random")," module), or your inference results may be invalid, since Bean Machine will not be aware of it."),(0,s.mdx)("p",null,"Calling random variable functions has different behaviors depending upon the callee's context, outlined in the next two sections."),(0,s.mdx)("h2",{id:"calling-a-random-variable-from-another-random-variable-function"},(0,s.mdx)("a",{name:"calling_inside"}),"Calling a Random Variable from Another Random Variable Function"),(0,s.mdx)("p",null,"When calling a random variable function from within another random variable function, you should treat the return value as a ",(0,s.mdx)("em",{parentName:"p"},"sample")," from its underlying distribution. Bean Machine intercepts these calls, and will perform inference-specific operations in order to draw a sample from the underlying distribution that is consistent with the available observation data. Working with samples therefore decouples your model definition from the mechanics of inference going on under the hood."),(0,s.mdx)("p",null,(0,s.mdx)("strong",{parentName:"p"},"Calls to random variable functions are effectively memoized during a particular inference iteration.")," This is a common pitfall, so it bears repeating: calls to the same random variable function with the same arguments will receive the same sampled value within one iteration of inference. This makes it easy for multiple components of your model to refer to the same logical random variable. This means that the common statistical notation discussed previously in ",(0,s.mdx)("a",{parentName:"p",href:"#declarative_style"},"Declarative Style")," can easily map to your code: a programmatic definition like ",(0,s.mdx)("inlineCode",{parentName:"p"},"reproduction_rate()")," will always map to its corresponding singular statistical concept of ",(0,s.mdx)("span",{parentName:"p",className:"math math-inline"},(0,s.mdx)("span",{parentName:"span",className:"katex"},(0,s.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,s.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,s.mdx)("semantics",{parentName:"math"},(0,s.mdx)("mrow",{parentName:"semantics"},(0,s.mdx)("msub",{parentName:"mrow"},(0,s.mdx)("mi",{parentName:"msub"},"n"),(0,s.mdx)("mtext",{parentName:"msub"},"new"))),(0,s.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"n_\\text{new}")))),(0,s.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.58056em",verticalAlign:"-0.15em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal"},"n"),(0,s.mdx)("span",{parentName:"span",className:"msupsub"},(0,s.mdx)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.151392em"}},(0,s.mdx)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,s.mdx)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,s.mdx)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord text mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord mtight"},"new"))))),(0,s.mdx)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,s.mdx)("span",{parentName:"span"})))))))))),", no matter how many times it is invoked within a single model. This can also be appreciated from a ",(0,s.mdx)("em",{parentName:"p"},"consistency")," point of view: if we define a new random variable ",(0,s.mdx)("inlineCode",{parentName:"p"},"tautology")," to be equal to ",(0,s.mdx)("inlineCode",{parentName:"p"},"reproduction_rate() <= 3.0 or reproduction_rate() > 3.0"),", the probability of ",(0,s.mdx)("inlineCode",{parentName:"p"},"tautology")," being ",(0,s.mdx)("inlineCode",{parentName:"p"},"True")," should be ",(0,s.mdx)("span",{parentName:"p",className:"math math-inline"},(0,s.mdx)("span",{parentName:"span",className:"katex"},(0,s.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,s.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,s.mdx)("semantics",{parentName:"math"},(0,s.mdx)("mrow",{parentName:"semantics"},(0,s.mdx)("mn",{parentName:"mrow"},"1")),(0,s.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"1")))),(0,s.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.64444em",verticalAlign:"0em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord"},"1"))))),", but if each invocation of ",(0,s.mdx)("inlineCode",{parentName:"p"},"reproduction_rate")," produced a different value, this would not hold."),(0,s.mdx)("p",null,"In ",(0,s.mdx)("a",{parentName:"p",href:"#random_variable_families"},"Defining Random Variable Families"),", we'll see how to control this memoization behavior with function parameters."),(0,s.mdx)("h2",{id:"calling-a-random-variable-from-an-ordinary-function"},(0,s.mdx)("a",{name:"calling_outside"}),"Calling a Random Variable from an Ordinary Function"),(0,s.mdx)("p",null,"It is valid to call random variable functions from ordinary Python functions. In fact, you've seen it a few times in the ",(0,s.mdx)("a",{parentName:"p",href:"/docs/quickstart"},"Quick Start")," already! We've used it to bind data, specify our queries, and access samples once inference has been completed."),(0,s.mdx)("p",null,"Under the hood, Bean Machine transforms random variable functions so that they act like function references. Here's an example, which we just call from the Python toplevel scope:"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-py"},"num_new_cases()\n")),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre"},"RVIdentifier(function=<function num_new_cases at 0x7ff00372d290>, arguments=())\n")),(0,s.mdx)("p",null,"As you can see, the call to this random variable function didn't return a distribution, or a sample from a distribution. Rather, it resulted in an ",(0,s.mdx)("inlineCode",{parentName:"p"},"RVIdentifier")," object, which represents a reference to a random variable function. You as the user can't do much with this object on its own, but Bean Machine will use this reference to access and re-evaluate different parts of your model."),(0,s.mdx)("h2",{id:"defining-random-variable-families"},(0,s.mdx)("a",{name:"random_variable_families"}),"Defining Random Variable Families"),(0,s.mdx)("p",null,"As discussed in ",(0,s.mdx)("a",{parentName:"p",href:"#calling_inside"},"Calling a Random Variable from Another Random Variable Function"),", calls to a random variable function are memoized during a particular iteration of inference. How, then, can we create models with many random variables which have related but distinct distributions?"),(0,s.mdx)("p",null,"Let's dive into this by extending our example model. In the previous example, we were modeling the number of new cases on a given day as a function of the number of infected individuals on the previous day. However, what if we wanted to  model the spread of disease over multiple days? This might correspond to the following mathematical model:"),(0,s.mdx)("ul",null,(0,s.mdx)("li",{parentName:"ul"},(0,s.mdx)("span",{parentName:"li",className:"math math-inline"},(0,s.mdx)("span",{parentName:"span",className:"katex"},(0,s.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,s.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,s.mdx)("semantics",{parentName:"math"},(0,s.mdx)("mrow",{parentName:"semantics"},(0,s.mdx)("msub",{parentName:"mrow"},(0,s.mdx)("mi",{parentName:"msub"},"n"),(0,s.mdx)("mi",{parentName:"msub"},"i")),(0,s.mdx)("mo",{parentName:"mrow"},"\u223c"),(0,s.mdx)("mtext",{parentName:"mrow"},"Poisson"),(0,s.mdx)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,s.mdx)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,s.mdx)("mn",{parentName:"mrow"},"1"),(0,s.mdx)("mo",{parentName:"mrow"},"+"),(0,s.mdx)("mtext",{parentName:"mrow",mathvariant:"monospace"},"reproduction_rate"),(0,s.mdx)("mo",{parentName:"mrow",stretchy:"false"},")"),(0,s.mdx)("mo",{parentName:"mrow"},"\u22c5"),(0,s.mdx)("msub",{parentName:"mrow"},(0,s.mdx)("mi",{parentName:"msub"},"n"),(0,s.mdx)("mrow",{parentName:"msub"},(0,s.mdx)("mi",{parentName:"mrow"},"i"),(0,s.mdx)("mo",{parentName:"mrow"},"\u2212"),(0,s.mdx)("mn",{parentName:"mrow"},"1"))),(0,s.mdx)("mo",{parentName:"mrow",stretchy:"false"},")")),(0,s.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"n_i \\sim \\text{Poisson}((1 + \\texttt{reproduction\\_rate}) \\cdot n_{i-1})")))),(0,s.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.58056em",verticalAlign:"-0.15em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal"},"n"),(0,s.mdx)("span",{parentName:"span",className:"msupsub"},(0,s.mdx)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.31166399999999994em"}},(0,s.mdx)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,s.mdx)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,s.mdx)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal mtight"},"i")))),(0,s.mdx)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,s.mdx)("span",{parentName:"span"})))))),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2777777777777778em"}}),(0,s.mdx)("span",{parentName:"span",className:"mrel"},"\u223c"),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2777777777777778em"}})),(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord text"},(0,s.mdx)("span",{parentName:"span",className:"mord"},"Poisson")),(0,s.mdx)("span",{parentName:"span",className:"mopen"},"("),(0,s.mdx)("span",{parentName:"span",className:"mopen"},"("),(0,s.mdx)("span",{parentName:"span",className:"mord"},"1"),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222222222222222em"}}),(0,s.mdx)("span",{parentName:"span",className:"mbin"},"+"),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222222222222222em"}})),(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord text"},(0,s.mdx)("span",{parentName:"span",className:"mord texttt"},"reproduction_rate")),(0,s.mdx)("span",{parentName:"span",className:"mclose"},")"),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222222222222222em"}}),(0,s.mdx)("span",{parentName:"span",className:"mbin"},"\u22c5"),(0,s.mdx)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2222222222222222em"}})),(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal"},"n"),(0,s.mdx)("span",{parentName:"span",className:"msupsub"},(0,s.mdx)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.311664em"}},(0,s.mdx)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,s.mdx)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,s.mdx)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal mtight"},"i"),(0,s.mdx)("span",{parentName:"span",className:"mbin mtight"},"\u2212"),(0,s.mdx)("span",{parentName:"span",className:"mord mtight"},"1"))))),(0,s.mdx)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.208331em"}},(0,s.mdx)("span",{parentName:"span"})))))),(0,s.mdx)("span",{parentName:"span",className:"mclose"},")"))))),", where ",(0,s.mdx)("span",{parentName:"li",className:"math math-inline"},(0,s.mdx)("span",{parentName:"span",className:"katex"},(0,s.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,s.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,s.mdx)("semantics",{parentName:"math"},(0,s.mdx)("mrow",{parentName:"semantics"},(0,s.mdx)("msub",{parentName:"mrow"},(0,s.mdx)("mi",{parentName:"msub"},"n"),(0,s.mdx)("mi",{parentName:"msub"},"i"))),(0,s.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"n_i")))),(0,s.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.58056em",verticalAlign:"-0.15em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal"},"n"),(0,s.mdx)("span",{parentName:"span",className:"msupsub"},(0,s.mdx)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.31166399999999994em"}},(0,s.mdx)("span",{parentName:"span",style:{top:"-2.5500000000000003em",marginLeft:"0em",marginRight:"0.05em"}},(0,s.mdx)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,s.mdx)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal mtight"},"i")))),(0,s.mdx)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,s.mdx)("span",{parentName:"span",className:"vlist-r"},(0,s.mdx)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,s.mdx)("span",{parentName:"span"}))))))))))," represents the number of cases on day ",(0,s.mdx)("span",{parentName:"li",className:"math math-inline"},(0,s.mdx)("span",{parentName:"span",className:"katex"},(0,s.mdx)("span",{parentName:"span",className:"katex-mathml"},(0,s.mdx)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,s.mdx)("semantics",{parentName:"math"},(0,s.mdx)("mrow",{parentName:"semantics"},(0,s.mdx)("mi",{parentName:"mrow"},"i")),(0,s.mdx)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"i")))),(0,s.mdx)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,s.mdx)("span",{parentName:"span",className:"base"},(0,s.mdx)("span",{parentName:"span",className:"strut",style:{height:"0.65952em",verticalAlign:"0em"}}),(0,s.mdx)("span",{parentName:"span",className:"mord mathnormal"},"i"))))),".")),(0,s.mdx)("p",null,"It is common for statistical models to group random variables together into a ",(0,s.mdx)("em",{parentName:"p"},"family")," of random variables as you see here."),(0,s.mdx)("p",null,"In Bean Machine, we generalize the ability to index into a family of random variables with arbitrary Python objects. We can extend our previous example to add an index onto our random variable ",(0,s.mdx)("inlineCode",{parentName:"p"},"num_new_cases()")," with an object of type ",(0,s.mdx)("inlineCode",{parentName:"p"},"datetime.date"),":"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-py"},"import datetime\n\n@bm.random_variable\ndef num_cases(day):\n    # Base case for recursion\n    if day == datetime.date(2021, 1, 1):\n        return dist.Poisson(num_infected)\n    return dist.Poisson(\n        (1 + reproduction_rate()) *  num_cases(day - datetime.timedelta(days=1))\n    )\n")),(0,s.mdx)("h2",{id:"transforming-random-variables"},"Transforming Random Variables"),(0,s.mdx)("p",null,"There's one last important construct in Bean Machine's modeling toolkit: ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.functional"),". This decorator is used to deterministically transform the results of one or more random variables."),(0,s.mdx)("p",null,"In the above example, you'll notice that we added 1 to the reproduction rate, to turn it into a coefficient for the previous day's number of cases. It would be nice to capture this as its own function. Here's an ",(0,s.mdx)("strong",{parentName:"p"},"incorrect")," attempt (don't do this!):"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-py"},"# COUNTER-EXAMPLE\n\ndef infection_rate():\n    return 1 + reproduction_rate()\n\n@bm.random_variable\ndef num_cases(day):\n    # Base case for recursion\n    if day == datetime.date(2021, 1, 1):\n        return dist.Poisson(num_infected)\n    return dist.Poisson(\n        infection_rate() *  num_cases(day - datetime.timedelta(days=1))\n    )\n")),(0,s.mdx)("p",null,"Why is this incorrect? You'll notice that ",(0,s.mdx)("inlineCode",{parentName:"p"},"num_cases()")," now calls into ",(0,s.mdx)("inlineCode",{parentName:"p"},"infection_rate()"),", which itself depends on the random variable function ",(0,s.mdx)("inlineCode",{parentName:"p"},"reproduction_rate()"),". We ",(0,s.mdx)("em",{parentName:"p"},"can't")," make ",(0,s.mdx)("inlineCode",{parentName:"p"},"infection_rate()")," a random variable function, as it does ",(0,s.mdx)("em",{parentName:"p"},"not")," return a ",(0,s.mdx)("a",{parentName:"p",href:"https://pytorch.org/docs/stable/distributions.html?highlight=distribution#module-torch.distributions"},"PyTorch distribution"),". However, since there is no ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.random_variable")," decorator, Bean Machine inference ",(0,s.mdx)("em",{parentName:"p"},"won't know")," that it should treat ",(0,s.mdx)("inlineCode",{parentName:"p"},"reproduction_rate()")," inside the function scope as a random variable function. Indeed, like we discussed in ",(0,s.mdx)("a",{parentName:"p",href:"#calling_outside"},"Calling a Random Variable from an Ordinary Function"),", ",(0,s.mdx)("inlineCode",{parentName:"p"},"reproduction_rate()")," in this context would merely return an ",(0,s.mdx)("inlineCode",{parentName:"p"},"RVIdentifier")," -- definitely not what we want."),(0,s.mdx)("p",null,"What do we do then? Bean Machine's ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.functional")," decorator is here to serve this exact purpose! ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.functional")," behaves like ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.random_variable")," except that it does not return a distribution. As such, it is appropriate to use to deterministically transform the results of one or more other ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.random_variable")," or ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.functional")," functions."),(0,s.mdx)("p",null,"Here's the correct way to write this model:"),(0,s.mdx)("pre",null,(0,s.mdx)("code",{parentName:"pre",className:"language-py"},"@bm.functional\ndef infection_rate():\n    return 1 + reproduction_rate()\n\n@bm.random_variable\ndef num_cases(day):\n    # Base case for recursion\n    if day == datetime.date(2021, 1, 1):\n        return dist.Poisson(num_infected)\n    return dist.Poisson(\n        infection_rate() *  num_cases(day - datetime.timedelta(days=1))\n    )\n")),(0,s.mdx)("p",null,"One last note: while a ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.functional")," can be queried (viewed) during inference, it can't be directly bound (softly constrained) to observations like a ",(0,s.mdx)("inlineCode",{parentName:"p"},"@bm.random_variable"),". This is because it is a deterministic function and thus inappropriate as a likelihood."),(0,s.mdx)("hr",null),(0,s.mdx)("p",null,"Next, we'll look at how you can use ",(0,s.mdx)("a",{parentName:"p",href:"/docs/overview/inference/"},"Inference")," to fit data to your model."))}c.isMDXComponent=!0}}]);