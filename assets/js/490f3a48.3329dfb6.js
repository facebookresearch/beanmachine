"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6353],{3905:function(e,n,a){a.r(n),a.d(n,{MDXContext:function(){return c},MDXProvider:function(){return h},mdx:function(){return b},useMDXComponents:function(){return u},withMDXComponents:function(){return d}});var i=a(67294);function t(e,n,a){return n in e?Object.defineProperty(e,n,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[n]=a,e}function o(){return o=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var a=arguments[n];for(var i in a)Object.prototype.hasOwnProperty.call(a,i)&&(e[i]=a[i])}return e},o.apply(this,arguments)}function r(e,n){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);n&&(i=i.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),a.push.apply(a,i)}return a}function s(e){for(var n=1;n<arguments.length;n++){var a=null!=arguments[n]?arguments[n]:{};n%2?r(Object(a),!0).forEach((function(n){t(e,n,a[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(a,n))}))}return e}function l(e,n){if(null==e)return{};var a,i,t=function(e,n){if(null==e)return{};var a,i,t={},o=Object.keys(e);for(i=0;i<o.length;i++)a=o[i],n.indexOf(a)>=0||(t[a]=e[a]);return t}(e,n);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(i=0;i<o.length;i++)a=o[i],n.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(t[a]=e[a])}return t}var c=i.createContext({}),d=function(e){return function(n){var a=u(n.components);return i.createElement(e,o({},n,{components:a}))}},u=function(e){var n=i.useContext(c),a=n;return e&&(a="function"==typeof e?e(n):s(s({},n),e)),a},h=function(e){var n=u(e.components);return i.createElement(c.Provider,{value:n},e.children)},m={inlineCode:"code",wrapper:function(e){var n=e.children;return i.createElement(i.Fragment,{},n)}},p=i.forwardRef((function(e,n){var a=e.components,t=e.mdxType,o=e.originalType,r=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),d=u(a),h=t,p=d["".concat(r,".").concat(h)]||d[h]||m[h]||o;return a?i.createElement(p,s(s({ref:n},c),{},{components:a})):i.createElement(p,s({ref:n},c))}));function b(e,n){var a=arguments,t=n&&n.mdxType;if("string"==typeof e||t){var o=a.length,r=new Array(o);r[0]=p;var s={};for(var l in n)hasOwnProperty.call(n,l)&&(s[l]=n[l]);s.originalType=e,s.mdxType="string"==typeof e?e:t,r[1]=s;for(var c=2;c<o;c++)r[c]=a[c];return i.createElement.apply(null,r)}return i.createElement.apply(null,a)}p.displayName="MDXCreateElement"},61687:function(e,n,a){a.r(n),a.d(n,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return c},toc:function(){return d},default:function(){return h}});var i=a(87462),t=a(63366),o=(a(67294),a(3905)),r=(a(44996),["components"]),s={slug:"/tutorials",title:"Tutorials",sidebar_label:"Tutorials"},l=void 0,c={unversionedId:"overview/tutorials/tutorials",id:"overview/tutorials/tutorials",title:"Tutorials",description:"These Bean Machine tutorials demonstrate various types of statistical models that users can build in Bean Machine.",source:"@site/../docs/overview/tutorials/tutorials.md",sourceDirName:"overview/tutorials",slug:"/tutorials",permalink:"/docs/tutorials",editUrl:"https://github.com/facebookresearch/beanmachine/edit/main/website/../docs/overview/tutorials/tutorials.md",tags:[],version:"current",frontMatter:{slug:"/tutorials",title:"Tutorials",sidebar_label:"Tutorials"},sidebar:"someSidebar",previous:{title:"Bean Machine Graph Inference",permalink:"/docs/beanstalk"},next:{title:"Contributing Docs",permalink:"/docs/contributing"}},d=[{value:"Tutorials",id:"tutorials",children:[{value:"Coin Flipping",id:"coin-flipping",children:[],level:3},{value:"Linear Regression",id:"linear-regression",children:[],level:3},{value:"Logistic Regression",id:"logistic-regression",children:[],level:3},{value:"Sparse Logistic Regression",id:"sparse-logistic-regression",children:[],level:3},{value:"Modeling Radon Decay Using a Hierarchical Regression with Continuous Data",id:"modeling-radon-decay-using-a-hierarchical-regression-with-continuous-data",children:[],level:3},{value:"Modeling MLB Performance Using Hierarchical Modeling with Repeated Binary Trials",id:"modeling-mlb-performance-using-hierarchical-modeling-with-repeated-binary-trials",children:[],level:3},{value:"Modeling NBA Foul Calls Using Item Response Theory",id:"modeling-nba-foul-calls-using-item-response-theory",children:[],level:3},{value:"Modeling Medical Efficacy by Marginalizing Discrete Variables in Zero-Inflated Count Data",id:"modeling-medical-efficacy-by-marginalizing-discrete-variables-in-zero-inflated-count-data",children:[],level:3},{value:"Hidden Markov Model",id:"hidden-markov-model",children:[],level:3},{value:"Gaussian Mixture Model",id:"gaussian-mixture-model",children:[],level:3},{value:"Neal&#39;s Funnel",id:"neals-funnel",children:[],level:3}],level:2}],u={toc:d};function h(e){var n=e.components,a=(0,t.Z)(e,r);return(0,o.mdx)("wrapper",(0,i.Z)({},u,a,{components:n,mdxType:"MDXLayout"}),(0,o.mdx)("p",null,"These Bean Machine tutorials demonstrate various types of statistical models that users can build in Bean Machine."),(0,o.mdx)("h2",{id:"tutorials"},"Tutorials"),(0,o.mdx)("h3",{id:"coin-flipping"},"Coin Flipping"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Coin_flipping.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Coin_flipping.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial demonstrates modeling and running inference on a simple coin-flipping model in Bean Machine. This should offer an accessible introduction to fundamental features of Bean Machine."),(0,o.mdx)("h3",{id:"linear-regression"},"Linear Regression"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Linear_Regression.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Linear_Regression.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial demonstrates modeling and running inference on a simple univariate linear regression model in Bean Machine. This should offer an accessible introduction to models that use PyTorch tensors and Newtonian Monte Carlo inference in Bean Machine. It will also teach you effective practices for prediction on new datasets with Bean Machine."),(0,o.mdx)("h3",{id:"logistic-regression"},"Logistic Regression"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Bayesian_Logistic_Regression.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Bayesian_Logistic_Regression.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial shows how to build a simple Bayesian model to deduce the line which separates two categories of points."),(0,o.mdx)("h3",{id:"sparse-logistic-regression"},"Sparse Logistic Regression"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Sparse_Logistic_Regression.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Sparse_Logistic_Regression.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial demonstrates modeling and running inference on a sparse logistic regression model in Bean Machine. This tutorial showcases the inference techniques in Bean Machine, and applies the model to a public dataset to evaluate performance. This tutorial will also introduce the ",(0,o.mdx)("inlineCode",{parentName:"p"},"@bm.functional")," decorator, which can be used to deterministically transform random variables. This tutorial uses it for convenient post-processing."),(0,o.mdx)("h3",{id:"modeling-radon-decay-using-a-hierarchical-regression-with-continuous-data"},"Modeling Radon Decay Using a Hierarchical Regression with Continuous Data"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Hierarchical_regression.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Hierarchical_regression.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial explores linear regression in combination with hierarchical priors. We will be using data from Gelman and Hill on radon levels found in buildings in Minnesota; ",(0,o.mdx)("a",{parentName:"p",href:"https://pdfs.semanticscholar.org/e6d6/8a23f02485cfbb3e35d6bba862a682a2f160.pdf"},"Hill J and Gelman A"),". This tutorial shows how  to prepare data for running a hierarchical regression with Bean Machine, how to run inference on that regression, and how to use ArviZ diagnostics to understand what Bean Machine is doing."),(0,o.mdx)("h3",{id:"modeling-mlb-performance-using-hierarchical-modeling-with-repeated-binary-trials"},"Modeling MLB Performance Using Hierarchical Modeling with Repeated Binary Trials"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Hierarchical_modeling.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Hierarchical_modeling.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial demonstrates the application of hierarchical models with data from the 1970 season of ",(0,o.mdx)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Major_League_Baseball"},"Major League Baseball (MLB)")," found in the paper by ",(0,o.mdx)("a",{parentName:"p",href:"http://www.medicine.mcgill.ca/epidemiology/hanley/bios602/MultilevelData/EfronMorrisJASA1975.pdf"},"Efron and Morris 1975"),". In addition to teaching effective hierarchical modeling techniques for binary data, this tutorial will explore how you can use different pooling techniques to enable strength-borrowing between observations."),(0,o.mdx)("h3",{id:"modeling-nba-foul-calls-using-item-response-theory"},"Modeling NBA Foul Calls Using Item Response Theory"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Item_Response_Theory.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Item_Response_Theory.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial demonstrates how to use Bean Machine to predict when NBA players will receive a foul call from a referee. This model and exposition is based on ",(0,o.mdx)("a",{parentName:"p",href:"https://austinrochford.com/posts/2018-02-04-nba-irt-2.html"},"Austin Rochford's 2018 analysis")," of the ",(0,o.mdx)("a",{parentName:"p",href:"https://www.basketball-reference.com/leagues/NBA_2016_games.html"},"2015/2016 NBA season games")," data. It will introduce you to Item Response Theory, and demonstrate its advantages over standard regression models. It will also iterate on that model several times, demonstrating how to evolve your model to improve predictive performance."),(0,o.mdx)("h3",{id:"modeling-medical-efficacy-by-marginalizing-discrete-variables-in-zero-inflated-count-data"},"Modeling Medical Efficacy by Marginalizing Discrete Variables in Zero-Inflated Count Data"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Zero_inflated_count_data.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Zero_inflated_count_data.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial investigates data that originated from ",(0,o.mdx)("a",{parentName:"p",href:"https://www.jstor.org/stable/2531826"},"Berry"),", and was analyzed by ",(0,o.mdx)("a",{parentName:"p",href:"https://www.jstor.org/stable/2531746"},"Farewell and Sprott"),", from a study about the efficacy of a medication that helps prevent irregular heartbeats. Counts of patients' irregular heartbeats were observed 60 seconds before the administration of the drug, and 60 seconds after the medication was taken. A large percentage of records show zero irregular heartbeats in the 60 seconds after taking the medication. There are more observed zeros than would be expected if we were to sample from one of the common statistical discrete distributions. The problem we face is trying to model these zero counts in order to appropriately quantify the medication's impact on reducing irregular heartbeats."),(0,o.mdx)("h3",{id:"hidden-markov-model"},"Hidden Markov Model"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Hidden_Markov_model.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Hidden_Markov_model.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial demonstrates modeling and running inference on a hidden Markov model (HMM) in Bean Machine. The flexibility of this model allows us to demonstrate useful features of Bean Machine, including ",(0,o.mdx)("inlineCode",{parentName:"p"},"CompositionalInference"),", multi-site inference, and posterior predictive checks. This model makes use of discrete latent states, and shows how Bean Machine can easily run inference for models comprised of both discrete and continuous latent variables."),(0,o.mdx)("h3",{id:"gaussian-mixture-model"},"Gaussian Mixture Model"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/GMM_with_2_dimensions_and_4_components.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/GMM_with_2_dimensions_and_4_components.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial uses Bean Machine to infer which latent clusters observed points are drawn from. It uses a 2-dimensional Gaussian mixture model with 4 mixture components, and shows how Bean Machine can automatically recover the means and variances of these latent components."),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Mixture_model"},"Mixture models")," are useful in problems where individuals from multiple sub-populations are aggregated together. A common use case for GMMs is unsupervised clustering, where one seeks to infer which sub-population an individual belongs without any labeled training data. Using Bean Machine, we provide a Bayesian treatment of this problem and infer a posterior distribution over cluster parameters and cluster assignments of observations."),(0,o.mdx)("h3",{id:"neals-funnel"},"Neal's Funnel"),(0,o.mdx)("p",null,(0,o.mdx)("a",{parentName:"p",href:"https://github.com/facebookresearch/beanmachine/blob/main/tutorials/Neals_funnel.ipynb"},"Open in GitHub")," \u2022 ",(0,o.mdx)("a",{parentName:"p",href:"https://colab.research.google.com/github/facebookresearch/beanmachine/blob/main/tutorials/Neals_funnel.ipynb"},"Run in Google Colab")),(0,o.mdx)("p",null,"This tutorial demonstrates modeling and running inference on the Neal's funnel model in Bean Machine. Neal's funnel is a synthetic model in which the posterior distribution is known beforehand, and Bean Machine's inference engine is tasked with automatically recovering that posterior distribution. Neal's funnel has proven difficult-to-handle for classical inference methods due to its unusual topology. This tutorial demonstrates how to overcome this by using second-order gradient methods in Bean Machine. It also demonstrates how to implement models with factors in Bean Machine through custom distributions."))}h.isMDXComponent=!0}}]);