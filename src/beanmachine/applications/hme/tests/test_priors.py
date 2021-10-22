# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import pandas as pd
import pytest
from beanmachine.applications.hme.abstract_linear_model import AbstractLinearModel
from beanmachine.applications.hme.abstract_model import AbstractModel
from beanmachine.applications.hme.configs import (
    InferConfig,
    ModelConfig,
    MixtureConfig,
    RegressionConfig,
    PriorConfig,
    StructuredPriorConfig,
)
from beanmachine.applications.hme.interface import HME
from beanmachine.applications.hme.priors import ParamType, Distribution


class RealizedModel(AbstractModel):
    def build_graph(self):
        return super().build_graph()


class RealizedLinearModel(AbstractLinearModel):
    def build_graph(self):
        self.fixed_effects, self.random_effects = self.parse_formula(
            self.model_config.mean_regression.formula
        )
        self._set_priors()
        self._set_default_priors()
        self._customize_priors()

    def _add_observation_byrow(self, index, row, fere_i):
        return super()._add_observation_byrow(index, row, fere_i)


@pytest.mark.parametrize(
    "label", ["unknown"],
)
def test_paramtype_exception(label):
    with pytest.raises(
        ValueError, match="Unknown parameter type: '{s}'".format(s=label)
    ):
        ParamType.match_str(label)


@pytest.mark.parametrize(
    "label", ["unknown"],
)
def test_distribution_exception(label):
    with pytest.raises(
        ValueError, match="Unknown distribution type: '{s}'".format(s=label)
    ):
        Distribution.match_str(label)


@pytest.mark.parametrize(
    "prior_config",
    [
        PriorConfig("normal", {"mean": 0.0, "sigma": 1.0}),
        PriorConfig("t", {"scale": 1.0, "mean": 0.0, "degree-of-freedom": 2.0}),
    ],
)
def test_parameter_dict_exception(prior_config):
    model = RealizedModel(data=None, model_config=ModelConfig())
    with pytest.raises(ValueError):
        model._parse_fe_prior_config(prior_config, "test")


@pytest.mark.parametrize(
    "const_value, const_type, expected_dot",
    [
        (1.0, "pos_real", 'digraph "graph" {\n  N0[label="1"];\n}\n'),
        (-1.0, "real", 'digraph "graph" {\n  N0[label="-1"];\n}\n'),
        (1, "natural", 'digraph "graph" {\n  N0[label="1"];\n}\n'),
        (0.5, "prob", 'digraph "graph" {\n  N0[label="0.5"];\n}\n'),
    ],
)
def test_generate_const_node(const_value, const_type, expected_dot):
    model = RealizedModel(data=None, model_config=ModelConfig())
    model._generate_const_node(const_value, ParamType.match_str(const_type))
    assert model.g.to_dot() == expected_dot


@pytest.mark.parametrize(
    "prior_config, expected_dot",
    [
        (
            PriorConfig("normal", {"scale": 1.0, "mean": 0.0}),
            """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="Normal"];
  N0 -> N2;
  N1 -> N2;
}
""",
        ),
        (
            PriorConfig("beta", {"alpha": 1.0, "beta": 1.0}),
            """
digraph "graph" {
  N0[label="1"];
  N1[label="1"];
  N2[label="Beta"];
  N0 -> N2;
  N1 -> N2;
}
""",
        ),
        (
            PriorConfig("gamma", {"beta": 1.0, "alpha": 1.0}),
            """
digraph "graph" {
  N0[label="1"];
  N1[label="1"];
  N2[label="Gamma"];
  N0 -> N2;
  N1 -> N2;
}
""",
        ),
        (
            PriorConfig("t", {"scale": 1.0, "mean": 0.0, "dof": 2.0}),
            """
digraph "graph" {
  N0[label="2"];
  N1[label="0"];
  N2[label="1"];
  N3[label="StudentT"];
  N0 -> N3;
  N1 -> N3;
  N2 -> N3;
}
""",
        ),
    ],
)
def test_parse_fe_prior_config(prior_config, expected_dot):
    model = RealizedModel(data=None, model_config=ModelConfig())
    model._parse_fe_prior_config(prior_config, "test")
    assert model.g.to_dot().strip() == expected_dot.strip()


def test_default_priors():
    model = RealizedLinearModel(data=None, model_config=ModelConfig(),)
    model._set_priors()
    model._set_default_priors()
    expected = {"fixed_effects": 8, "prob_h": 4, "prob_sign": 4}
    assert model.default_priors == expected


@pytest.mark.parametrize(
    "priors_desc, expected",
    [
        (
            {
                "x1": PriorConfig("normal", {"mean": 0.0, "scale": 1.0}),
                "x2": PriorConfig("flat", {}),
                "group": PriorConfig(
                    "t",
                    {
                        "mean": PriorConfig("normal", {"mean": 2.0, "scale": 1.0}),
                        "dof": PriorConfig("half_cauchy", {"scale": 1.0}),
                        "scale": 1.0,
                    },
                ),
                "prob_h": PriorConfig("beta", {"beta": 1.0, "alpha": 1.0}),
                "prob_sign": PriorConfig("beta", {"alpha": 1.0, "beta": 1.0}),
            },
            {
                "x1": 12,
                "x2": 13,
                "group": PriorConfig(
                    distribution="t",
                    parameters={
                        "mean": PriorConfig(
                            distribution="normal",
                            parameters={"mean": 2.0, "scale": 1.0},
                        ),
                        "dof": PriorConfig(
                            distribution="half_cauchy", parameters={"scale": 1.0}
                        ),
                        "scale": 1.0,
                    },
                ),
                "prob_h": 16,
                "prob_sign": 19,
            },
        ),
    ],
)
def test_customize_priors(priors_desc, expected):
    model = RealizedLinearModel(
        data=None,
        model_config=ModelConfig(
            mean_regression=RegressionConfig(formula="y~1"), priors=priors_desc,
        ),
    )
    model.build_graph()
    assert model.customized_priors == expected


@pytest.mark.parametrize(
    "priors_desc, expected_dot",
    [
        (
            {
                "x1": PriorConfig("t", {"dof": 2.0, "scale": 1.0, "mean": 0.0}),
                "x2": PriorConfig("beta", {"alpha": 1.0, "beta": 1.0}),
            },
            """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="2"];
  N3[label="3"];
  N4[label="Beta"];
  N5[label="Gamma"];
  N6[label="HalfCauchy"];
  N7[label="HalfNormal"];
  N8[label="Normal"];
  N9[label="StudentT"];
  N10[label="2"];
  N11[label="0"];
  N12[label="1"];
  N13[label="StudentT"];
  N14[label="1"];
  N15[label="1"];
  N16[label="Beta"];
  N17[label="~"];
  N18[label="~"];
  N0 -> N8;
  N0 -> N9;
  N1 -> N4;
  N1 -> N4;
  N1 -> N5;
  N1 -> N5;
  N1 -> N6;
  N1 -> N7;
  N2 -> N8;
  N3 -> N9;
  N3 -> N9;
  N10 -> N13;
  N11 -> N13;
  N12 -> N13;
  N13 -> N17;
  N14 -> N16;
  N15 -> N16;
  N16 -> N18;
}
""",
        ),
        (
            {},
            """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="2"];
  N3[label="3"];
  N4[label="Beta"];
  N5[label="Gamma"];
  N6[label="HalfCauchy"];
  N7[label="HalfNormal"];
  N8[label="Normal"];
  N9[label="StudentT"];
  N10[label="~"];
  N11[label="~"];
  N0 -> N8;
  N0 -> N9;
  N1 -> N4;
  N1 -> N4;
  N1 -> N5;
  N1 -> N5;
  N1 -> N6;
  N1 -> N7;
  N2 -> N8;
  N3 -> N9;
  N3 -> N9;
  N8 -> N10;
  N8 -> N11;
}
""",
        ),
    ],
)
def test_initialize_fixed_effect_nodes(priors_desc, expected_dot):
    model = RealizedLinearModel(
        data=None,
        model_config=ModelConfig(
            mean_regression=RegressionConfig(formula="y~x1+x2"), priors=priors_desc,
        ),
    )
    model.build_graph()
    model._initialize_fixed_effect_nodes()
    assert model.g.to_dot().strip() == expected_dot.strip()


@pytest.mark.parametrize(
    "priors_desc, expected_dot",
    [
        (
            {
                "group": PriorConfig(
                    "t",
                    {
                        "mean": PriorConfig("normal", {"mean": 2.0, "scale": 1.0}),
                        "dof": PriorConfig("half_normal", {"scale": 1.0}),
                        "scale": 1.0,
                    },
                ),
            },
            """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="2"];
  N3[label="3"];
  N4[label="Beta"];
  N5[label="Gamma"];
  N6[label="HalfCauchy"];
  N7[label="HalfNormal"];
  N8[label="Normal"];
  N9[label="StudentT"];
  N10[label="1"];
  N11[label="HalfNormal"];
  N12[label="~"];
  N13[label="2"];
  N14[label="1"];
  N15[label="Normal"];
  N16[label="~"];
  N17[label="1"];
  N18[label="StudentT"];
  N0 -> N8;
  N0 -> N9;
  N1 -> N4;
  N1 -> N4;
  N1 -> N5;
  N1 -> N5;
  N1 -> N6;
  N1 -> N7;
  N2 -> N8;
  N3 -> N9;
  N3 -> N9;
  N10 -> N11;
  N11 -> N12;
  N12 -> N18;
  N13 -> N15;
  N14 -> N15;
  N15 -> N16;
  N16 -> N18;
  N17 -> N18;
}
""",
        ),
        (
            {},
            """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="2"];
  N3[label="3"];
  N4[label="Beta"];
  N5[label="Gamma"];
  N6[label="HalfCauchy"];
  N7[label="HalfNormal"];
  N8[label="Normal"];
  N9[label="StudentT"];
  N10[label="~"];
  N11[label="Normal"];
  N0 -> N8;
  N0 -> N9;
  N0 -> N11;
  N1 -> N4;
  N1 -> N4;
  N1 -> N5;
  N1 -> N5;
  N1 -> N6;
  N1 -> N7;
  N2 -> N8;
  N3 -> N9;
  N3 -> N9;
  N7 -> N10;
  N10 -> N11;
}
""",
        ),
    ],
)
def test_initialize_random_effect_nodes(priors_desc, expected_dot):
    model = RealizedLinearModel(
        data=None,
        model_config=ModelConfig(
            mean_regression=RegressionConfig(formula="y~1+(1|group)"),
            priors=priors_desc,
        ),
    )
    model.build_graph()
    model._initialize_random_effect_nodes()
    assert model.g.to_dot().strip() == expected_dot.strip()


@pytest.mark.parametrize(
    "priors_desc, expected_dot",
    [
        (
            {},
            """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="2"];
  N3[label="3"];
  N4[label="Beta"];
  N5[label="Gamma"];
  N6[label="HalfCauchy"];
  N7[label="HalfNormal"];
  N8[label="Normal"];
  N9[label="StudentT"];
  N10[label="~"];
  N11[label="Normal"];
  N12[label="~"];
  N13[label="Normal"];
  N0 -> N8;
  N0 -> N9;
  N0 -> N11;
  N0 -> N13;
  N1 -> N4;
  N1 -> N4;
  N1 -> N5;
  N1 -> N5;
  N1 -> N6;
  N1 -> N7;
  N2 -> N8;
  N3 -> N9;
  N3 -> N9;
  N7 -> N10;
  N7 -> N12;
  N10 -> N11;
  N12 -> N13;
}
""",
        ),
    ],
)
def test_iid_random_effect_nodes(priors_desc, expected_dot):
    model = RealizedLinearModel(
        data=None,
        model_config=ModelConfig(
            mean_regression=RegressionConfig(formula="y~1+(1|group)"),
            priors=priors_desc,
        ),
    )
    model.build_graph()
    # generate two separate (i.i.d.) sample nodes from the same hyper-prior
    model._initialize_random_effect_nodes()
    model._initialize_random_effect_nodes()
    assert model.g.to_dot().strip() == expected_dot.strip()


def test_initialize_AR_structured_priors():
    priors_desc = {
        "age": StructuredPriorConfig(
            specification="AR", category_order=["Young", "Middle", "Elderly"]
        ),
    }

    model = RealizedLinearModel(
        data=None,
        model_config=ModelConfig(
            mean_regression=RegressionConfig(formula="y~1+(1|age)"), priors=priors_desc,
        ),
    )
    model.build_graph()
    # structured priors can only be defined for random effects
    model._initialize_random_effect_nodes()

    expected_dot = """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="2"];
  N3[label="3"];
  N4[label="Beta"];
  N5[label="Gamma"];
  N6[label="HalfCauchy"];
  N7[label="HalfNormal"];
  N8[label="Normal"];
  N9[label="StudentT"];
  N10[label="~"];
  N11[label="0.5"];
  N12[label="Beta"];
  N13[label="~"];
  N14[label="ToReal"];
  N15[label="-1"];
  N16[label="2"];
  N17[label="*"];
  N18[label="+"];
  N19[label="1"];
  N20[label="*"];
  N21[label="Negate"];
  N22[label="+"];
  N23[label="-0.5"];
  N24[label="^"];
  N25[label="ToPosReal"];
  N26[label="*"];
  N27[label="Normal"];
  N28[label="~"];
  N29[label="*"];
  N30[label="Normal"];
  N31[label="~"];
  N32[label="*"];
  N33[label="Normal"];
  N34[label="~"];
  N0 -> N8;
  N0 -> N9;
  N0 -> N27;
  N1 -> N4;
  N1 -> N4;
  N1 -> N5;
  N1 -> N5;
  N1 -> N6;
  N1 -> N7;
  N2 -> N8;
  N3 -> N9;
  N3 -> N9;
  N7 -> N10;
  N10 -> N26;
  N10 -> N30;
  N10 -> N33;
  N11 -> N12;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
  N14 -> N17;
  N15 -> N18;
  N16 -> N17;
  N17 -> N18;
  N18 -> N20;
  N18 -> N20;
  N18 -> N29;
  N18 -> N32;
  N19 -> N22;
  N20 -> N21;
  N21 -> N22;
  N22 -> N24;
  N23 -> N24;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
  N27 -> N28;
  N28 -> N29;
  N29 -> N30;
  N30 -> N31;
  N31 -> N32;
  N32 -> N33;
  N33 -> N34;
}
"""
    assert model.g.to_dot().strip() == expected_dot.strip()


def test_initialize_RW_structured_priors():
    priors_desc = {
        "age": StructuredPriorConfig(
            specification="RW", category_order=["Young", "Middle", "Elderly"]
        ),
    }

    model = RealizedLinearModel(
        data=None,
        model_config=ModelConfig(
            mean_regression=RegressionConfig(formula="y~1+(1|age)"), priors=priors_desc,
        ),
    )
    model.build_graph()
    # structured priors can only be defined for random effects
    model._initialize_random_effect_nodes()

    expected_dot = """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="2"];
  N3[label="3"];
  N4[label="Beta"];
  N5[label="Gamma"];
  N6[label="HalfCauchy"];
  N7[label="HalfNormal"];
  N8[label="Normal"];
  N9[label="StudentT"];
  N10[label="~"];
  N11[label="Flat"];
  N12[label="~"];
  N13[label="Normal"];
  N14[label="~"];
  N15[label="+"];
  N16[label="Negate"];
  N17[label="0.03"];
  N18[label="Normal"];
  N19[label="~"];
  N0 -> N8;
  N0 -> N9;
  N1 -> N4;
  N1 -> N4;
  N1 -> N5;
  N1 -> N5;
  N1 -> N6;
  N1 -> N7;
  N2 -> N8;
  N3 -> N9;
  N3 -> N9;
  N7 -> N10;
  N10 -> N13;
  N11 -> N12;
  N12 -> N13;
  N12 -> N15;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N18;
  N17 -> N18;
  N18 -> N19;
}
"""
    assert model.g.to_dot().strip() == expected_dot.strip()


@pytest.mark.parametrize(
    "mean_config, mixture_config, priors_desc, data, expected",
    [
        (
            RegressionConfig(
                distribution="normal",
                outcome="y",
                stderr="se",
                formula="~ 1+x1+x2+(1|age)+(1|group)",
                link="identity",
            ),
            MixtureConfig(use_null_mixture=True),
            {
                "x1": PriorConfig("normal", {"mean": 0.0, "scale": 1.0}),
                "group": PriorConfig(
                    "t",
                    {
                        "scale": 1.0,
                        "dof": PriorConfig("half_cauchy", {"scale": 1.0}),
                        "mean": PriorConfig("normal", {"mean": 2.0, "scale": 1.0}),
                    },
                ),
                "age": StructuredPriorConfig("AR", ["young", "middle", "elderly"]),
            },
            pd.DataFrame(
                {
                    "y": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "x1": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                    "x2": [2.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                    "se": [0.15] * 6,
                    "group": ["a"] * 3 + ["b"] * 3,
                    "team": ["x", "y"] * 3,
                    "age": ["young", "middle", "elderly"] * 2,
                }
            ),
            {
                "fixed_effect_Intercept",
                "fixed_effect_x1",
                "fixed_effect_x2",
                "re_age_rho",
                "re_age_sigma",
                "re_group_dof",
                "re_group_mean",
                "re_value_age_young",
                "re_value_age_middle",
                "re_value_age_elderly",
                "re_value_group_a",
                "re_value_group_b",
            },
        ),
    ],
)
def test_queries(mean_config, mixture_config, priors_desc, data, expected):
    model = HME(
        data,
        ModelConfig(
            mean_regression=mean_config, mean_mixture=mixture_config, priors=priors_desc
        ),
    )
    post_samples, _ = model.infer(InferConfig(n_iter=100, n_warmup=100, seed=0))
    assert expected.issubset(post_samples.columns)
