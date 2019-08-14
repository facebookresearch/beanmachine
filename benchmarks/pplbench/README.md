# Readme: Getting Started with PPLBench

## What is PPLBench?

PPLBench is a benchmarking tool for analyzing the performance of various PPLs in context of their ability to implement one or more of the popular Bayesian models. It is designed to be modular so new models and PPL implementations of models can be added into this framework. The fundamental evaluation metric is the average log predictive log likelihood on test set, though more model specific evaluations could be added later.

## How to use PPLBench?

### Installation:

Following is the procedure to install PPLBench on Linux (Tested on RHEL/CentOS):

1. Download/Clone PPLBench [https://github.com/facebookincubator/BeanMachine/]
2. Installing dependencies:
    1. PPLBench core:
        `pip install -r requirements.txt`
    2. PPLs (Only need to install the ones which you want to benchmark):
        1. Stan:
            `pip install pystan`
        2. Jags:
            see Appendix I below
        3. pymc3:
            `pip install pymc3`
        4. pyro:
            `pip install pyro-ppl`

### Example:

Let us go through an example to check if the installation is working. From the PPLBench directory, run the following command:

```
python PPLBench.py -m robustRegression -l jags,stan  -k 5 -n 2000 -t 30 --iterations 2
```

this should take around 2-4 mins to complete and should produce a result similar to this:

![Test output](docs/readme_test_output.png)

If the plot looks similar, you’re all set!

### Next steps:

* For more information you can look up SPECIFICATIONS.md in the docs folder.
* To see supported models, PPL implementations, usage and optional arguments, type:

```
python PPLBench.py -h
```

### Appendix I: Installing Jags and pyJags:

Installing Jags and PyJags is not as easy as other PPLs. Here is the procedure:

****Step 1: Install JAGS from source****

1. Download JAGS from source forge:

    `wget [https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Source/JAGS-4.3.0.tar.gz](https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Source/JAGS-4.3.0.tar.gz)`
2. Extract:

    `tar xvzf JAGS-4.3.0.tar.gz`
3. Make:

    `cd JAGS-4.3.0`

    `./configure make -j4`

    `make install`

    note: you may need sudo privilege for this step

****Step 2: Installing PyJAGS****

1. Use pip:

    `pip install pyjags`
2. Build from source:

    `git clone [https://github.com/SourabhKul/pyjags](https://github.com/SourabhKul/pyjags)`

    `cd pyjags`

    `pip install .`

****Step 3: Installing JAGS from rpm file (**only** **if** JAGS built from source does not work; it probably won’t)****

1. Download the latest RPM file:

    `wget [http://download.opensuse.org/repositories/home:/cornell_vrdc/CentOS_7/x86_64/jags4-4.3.0-67.4.x86_64.rpm](http://download.opensuse.org/repositories/home:/cornell_vrdc/CentOS_7/x86_64/jags4-4.3.0-67.4.x86_64.rpm)`
2. Install:

    `sudo rpm -i jags4-4.3.0-67.4.x86_64.rpm`
