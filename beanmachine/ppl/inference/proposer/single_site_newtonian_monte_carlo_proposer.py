# Copyright (c) Facebook, Inc. and its affiliates
from typing import Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world.variable import Variable
from torch import Tensor
from torch.autograd import grad


class SingleSiteNewtonianMonteCarloProposer(SingleSiteAncestralProposer):
    """
    Single-Site Newtonian Monte Carlo Implementations

    In this implementation, we draw a new sample from a proposal that is a
    MultivariateNormal with followings specifications:
        mean = sampled_value - gradient * hessian_inversed
        covariance = - hessian_inversed

    Where the gradient and hessian are computed over the log probability of
    the node being resampled and log probabilities of its immediate children

    To compute, the proposal log update, we go through the following steps:
        1) Draw a new sample from MultivariateNormal(node.mean, node.covariance)
        and compute the log probability of the new draw coming from this
        proposal distribution log(P(X->X'))
        2) Construct the new diff given the new value
        3) Compute new gradient and hessian and hence, new mean and
        covariance and compute the log probability of the old value coming
        from MutlivariateNormal(new mean, new covariance) log(P(X'->X))
        4) Compute the final proposal log update: log(P(X'->X)) - log(P(X->X'))
    """

    def __init__(self, world):
        super().__init__(world)

    def is_valid(self, vec: Tensor) -> bool:
        """
        :returns: whether a tensor is valid or not (not nan and not inf)
        """
        return not (torch.isnan(vec).any() or torch.isinf(vec).any())

    def zero_grad(self, node_val: Tensor):
        """
        Zeros the gradient.
        """
        if hasattr(node_val, "grad") and node_val.grad is not None:
            node_val.grad.zero_()

    def compute_score(self, node_var: Variable) -> Tensor:
        """
        Computes the score of the node plus its children

        :param node_var: the node variable whose score we are going to compute
        :returns: the computed score
        """
        score = node_var.log_prob.clone()
        for child in node_var.children:
            score += self.world_.get_node_in_world(child, False).log_prob.clone()
        return score

    def compute_first_gradient(
        self, score: Tensor, node_val: Tensor
    ) -> Tuple[bool, Tensor]:
        """
        Computes the first gradient.

        :param score: the score to compute the gradient of
        :param node_val: the value to compute the gradient against
        :returns: the first gradient
        """
        if not hasattr(node_val, "grad"):
            raise ValueError("requires_grad_ needs to be set for node values")

        # pyre expects attributes to be defined in constructors or at class
        # top levels and doesn't support attributes that get dynamically added.
        # pyre-fixme
        elif node_val.grad is not None:
            node_val.grad.zero_()

        first_gradient = grad(score, node_val, create_graph=True)[0]
        return self.is_valid(first_gradient), first_gradient

    def compute_hessian(
        self, first_gradient: Tensor, node_val: Tensor
    ) -> Tuple[bool, Tensor]:
        """
        Computes the hessian

        :param first_gradient: the first gradient of score with respect to
        node_val
        :param node_val: the value to compute the hessian against
        :returns: computes hessian
        """
        hessian = None
        size = first_gradient.shape[0]
        for i in range(size):

            second_gradient = (
                grad(
                    # pyre-fixme
                    first_gradient.index_select(0, tensor([i])),
                    node_val,
                    create_graph=True,
                )[0]
            ).reshape(-1)

            hessian = (
                torch.cat((hessian, (second_gradient).unsqueeze(0)), 0)
                if hessian is not None
                else (second_gradient).unsqueeze(0)
            )
        if hessian is None:
            raise ValueError("Something went wrong with gradient computation")

        if not self.is_valid(hessian):
            return False, tensor(0.0)

        return True, hessian

    def compute_neg_hessian_invserse(
        self, first_gradient: Tensor, node_val: Tensor
    ) -> Tuple[bool, Tensor]:
        """
        Compute negative hessian inverse.

        :param first_gradient: the first gradient of score with respect to
        node_val
        :param node_val: the value to compute the hessian against
        :returns: computes negative hessian inverse
        """
        is_valid, hessian = self.compute_hessian(first_gradient, node_val)
        if not is_valid:
            return False, tensor(0.0)
        # to avoid problems with inverse, here we add a small value - 1e-7 to
        # the diagonals
        diag = (1e-7) * torch.eye(hessian.shape[0])
        # pyre-fixme
        hessian_inverse = (hessian + diag).inverse()
        hessian_inverse = (hessian_inverse + hessian_inverse.T) / 2
        neg_hessian_inverse = -1 * hessian_inverse
        eig_vals, eig_vec = torch.eig(neg_hessian_inverse, eigenvectors=True)
        eig_vals = eig_vals[:, 0]
        num_neg_eig_vals = (eig_vals < 0).sum()
        if num_neg_eig_vals.item() > 0:
            eig_vals[eig_vals < 0] = 1e-5
            eig_vals = torch.eye(len(eig_vals)) * eig_vals
            eig_vals_64 = eig_vals.to(dtype=torch.float64)
            eig_vec_64 = eig_vec.to(dtype=torch.float64)
            neg_hessian_inverse = eig_vec_64 @ eig_vals_64 @ eig_vec_64.T
            if eig_vals.dtype is torch.float32:
                neg_hessian_inverse = neg_hessian_inverse.to(dtype=torch.float32)

        return True, neg_hessian_inverse

    def compute_normal_mean_covar(
        self, node_var: Variable
    ) -> Tuple[bool, Tensor, Tensor]:
        """
        Computes mean and covariance of the MultivariateNormal given the node.

        :param node_var: the node Variable we're proposing a new value for
        :returns: mean and covariance
        """
        node_val = node_var.value
        score = self.compute_score(node_var)
        self.zero_grad(node_val)
        is_valid_gradient, gradient = self.compute_first_gradient(score, node_val)

        if not is_valid_gradient:
            self.zero_grad(node_val)
            return False, tensor(0.0), tensor(0.0)

        first_gradient = gradient.reshape(-1).clone()
        is_valid_neg_invserse_hessian, neg_hessian_inverse = self.compute_neg_hessian_invserse(
            first_gradient, node_val
        )
        self.zero_grad(node_val)
        if not is_valid_neg_invserse_hessian:
            return False, tensor(0.0), tensor(0.0)

        # node value may of any arbitrary shape, so here, we transform it into a
        # 1D vector using reshape(-1) and with unsqueeze(0), we change 1D vector
        # of size (N) to (1 x N) matrix.
        node_reshaped = node_val.reshape(-1).unsqueeze(0)
        # here we again call unsqueeze(0) on first_gradient to transform it into
        # a matrix in order to be able to perform matrix multiplication.
        mean = (
            node_reshaped
            # pyre-fixme
            + first_gradient.unsqueeze(0).mm(neg_hessian_inverse)
        ).squeeze(0)
        covariance = neg_hessian_inverse
        return True, mean, covariance

    def compute_alpha_beta(self, node_var: Variable) -> Tuple[bool, Tensor, Tensor]:
        """
        Computes alpha and beta of the Gamma proposal given the node.

        :param node_var: the node Variable we're proposing a new value for
        :returns: alpha and beta of the Gamma distribution as proposal
        distribution
        """
        node_val = node_var.value
        score = self.compute_score(node_var)
        self.zero_grad(node_val)
        is_valid_gradient, gradient = self.compute_first_gradient(score, node_val)
        if not is_valid_gradient:
            self.zero_grad(node_val)
            return False, tensor(0.0), tensor(0.0)
        first_gradient = gradient.reshape(-1).clone()
        is_valid_hessian, hessian = self.compute_hessian(first_gradient, node_val)
        self.zero_grad(node_val)
        if not is_valid_hessian:
            return False, tensor(0.0), tensor(0.0)

        # pyre-fixme
        hessian_diag = hessian.diag()
        # pyre-fixme
        predicted_alpha = (1 - hessian_diag * (node_val * node_val)).T
        predicted_beta = -1 * node_val * hessian_diag - first_gradient
        condition = (predicted_alpha > 0) & (predicted_beta > 0)
        predicted_alpha = torch.where(condition, predicted_alpha, tensor(1.0))
        predicted_beta = torch.where(
            condition,
            predicted_beta,
            # pyre-fixme
            tensor(1.0) / node_var.distribution.mean,
        )
        predicted_alpha.reshape(node_val.shape)
        predicted_beta.reshape(node_val.shape)
        return True, predicted_alpha, predicted_beta

    def compute_alpha(self, node_var: Variable) -> Tuple[bool, Tensor]:
        """
        Computes alpha of the Dirichlet proposal given the node.

        :param node_var: the node Variable we're proposing a new value for
        :returns: alpha of the Dirichlet distribution as proposal distribution
        """
        node_val = (
            node_var.value if node_var.extended_val is None else node_var.extended_val
        )
        score = self.compute_score(node_var)
        is_valid_gradient, gradient = self.compute_first_gradient(score, node_val)
        if not is_valid_gradient:
            self.zero_grad(node_val)
            return False, tensor(0.0)

        first_gradient = gradient.clone().reshape(-1)
        size = first_gradient.shape[0]
        hessian_diag_minus_max = torch.zeros(size)
        for i in range(size):
            second_gradient = (
                grad(
                    # pyre-fixme
                    first_gradient.index_select(0, tensor([i])),
                    node_val,
                    create_graph=True,
                )[0]
            ).reshape(-1)

            if not self.is_valid(second_gradient):
                return False, tensor(0.0)

            hessian_diag_minus_max[i] = second_gradient[i]
            second_gradient[i] = 0
            hessian_diag_minus_max[i] -= second_gradient.max()

        self.zero_grad(node_val)

        node_val_reshaped = node_val.clone().reshape(-1)
        predicted_alpha = (
            1 - ((node_val_reshaped * node_val_reshaped) * (hessian_diag_minus_max))
        ).reshape(node_val.shape)
        predicted_alpha = torch.where(
            predicted_alpha < -1 * 1e-3,
            # pyre-fixme
            node_var.distribution.mean,
            predicted_alpha,
        )

        predicted_alpha = torch.where(
            predicted_alpha > 0, predicted_alpha, tensor(1e-3)
        )
        return True, predicted_alpha

    def post_process(self, node: RVIdentifier) -> Tensor:
        """
        Computes new gradient, with the new proposal distribution and finally
        computes the log probability of the old value coming from the proposal
        distribution.

        :param node: the node for which we have already proposed a new value for.
        :returns: the log probability of proposing the old value from this new world.
        """
        node_var = self.world_.get_node_in_world(node, False)
        support = node_var.distribution.support

        if isinstance(support, dist.constraints._Real):
            is_valid, old_value_log_proposal = self.post_process_for_real_support(
                node, node_var
            )

        elif isinstance(support, dist.constraints._Simplex) or isinstance(
            node_var.distribution, dist.Beta
        ):
            is_valid, old_value_log_proposal = self.post_process_for_simplex_support(
                node, node_var
            )
        elif (
            isinstance(support, dist.constraints._GreaterThan)
            and support.lower_bound == 0
        ):
            is_valid, old_value_log_proposal = self.post_process_for_halfspace_support(
                node, node_var
            )
        else:
            return super().post_process(node)

        if is_valid:
            return old_value_log_proposal
        return super().post_process(node)

    def post_process_for_simplex_support(
        self, node: RVIdentifier, node_var: Variable
    ) -> Tuple[bool, Tensor]:
        """
        Computes new gradient, with alpha for the Dirichlet proposal
        distribution and finally computes the log probability of the old value
        coming from the Dirichlet distribution.

        :param node: the node for which we have already proposed a new value for.
        :returns: the log probability of proposing the old value from this new world.
        """
        old_value = (
            self.world_.variables_[node].extended_val
            if isinstance(node_var.distribution, dist.Beta)
            else self.world_.variables_[node].value
        )
        number_of_variables = len(self.world_.variables_) - len(
            self.world_.observations_
        )
        proposal_distribution = node_var.proposal_distribution
        if proposal_distribution is None or number_of_variables != 1:
            is_valid, alpha = self.compute_alpha(node_var)
            if not is_valid:
                return False, tensor(0.0)
            proposal_distribution = dist.Dirichlet(alpha)
            node_var.proposal_distribution = proposal_distribution
        # pyre-fixme
        return True, proposal_distribution.log_prob(old_value).sum()

    def post_process_for_real_support(
        self, node: RVIdentifier, node_var: Variable
    ) -> Tuple[bool, Tensor]:
        """
        Computes new gradient and hessian and hence, new mean and covariance and
        finally computes the log probability of the old value coming from
        MutlivariateNormal(new mean, new covariance).

        :param node: the node for which we have already proposed a new value for.
        :returns: the log probability of proposing the old value from this new world.
        """
        old_value = self.world_.variables_[node].value
        number_of_variables = len(self.world_.variables_) - len(
            self.world_.observations_
        )
        proposal_distribution = node_var.proposal_distribution
        if proposal_distribution is None or number_of_variables != 1:
            is_valid, mean, covariance = self.compute_normal_mean_covar(node_var)
            if not is_valid:
                return False, tensor(0.0)
            proposal_distribution = dist.MultivariateNormal(mean, covariance)
            node_var.proposal_distribution = proposal_distribution
        old_value = old_value.reshape(-1)
        # pyre-fixme
        return True, proposal_distribution.log_prob(old_value).sum()

    def post_process_for_halfspace_support(
        self, node: RVIdentifier, node_var: Variable
    ) -> Tuple[bool, Tensor]:
        """
        Computes new gradient and hessian and hence, new mean and covariance and
        finally computes the log probability of the old value coming from
        MutlivariateNormal(new mean, new covariance).

        :param node: the node for which we have already proposed a new value for.
        :returns: the log probability of proposing the old value from this new world.
        """
        old_value = self.world_.variables_[node].value
        number_of_variables = len(self.world_.variables_) - len(
            self.world_.observations_
        )
        proposal_distribution = node_var.proposal_distribution
        if proposal_distribution is None or number_of_variables != 1:
            is_valid, alpha, beta = self.compute_alpha_beta(node_var)
            if not is_valid:
                return False, tensor(0.0)
            proposal_distribution = dist.Gamma(alpha, beta)
            node_var.proposal_distribution = proposal_distribution
        old_value = old_value.reshape(-1)
        # pyre-fixme
        return True, proposal_distribution.log_prob(old_value).sum()

    def propose(self, node: RVIdentifier) -> Tuple[Tensor, Tensor]:
        """
        Proposes a new value for the node by drawing a sample from the proposal
        distribution and compute the log probability of the new draw coming from
        this proposal distribution log(P(X->X')).

        :param node: the node for which we'll need to propose a new value for.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        node_var = self.world_.get_node_in_world(node, False)

        support = node_var.distribution.support

        if isinstance(support, dist.constraints._Real):
            is_valid, proposed_value, negative_new_value_log_proposal = self.propose_for_real_support(
                node_var
            )

        elif isinstance(support, dist.constraints._Simplex) or isinstance(
            node_var.distribution, dist.Beta
        ):
            is_valid, proposed_value, negative_new_value_log_proposal = self.propose_for_simplex_support(
                node_var
            )
        elif (
            isinstance(support, dist.constraints._GreaterThan)
            and support.lower_bound == 0
        ):
            is_valid, proposed_value, negative_new_value_log_proposal = self.propose_for_hspace_support(
                node_var
            )
        else:
            return super().propose(node)

        if is_valid:
            return proposed_value, negative_new_value_log_proposal
        return super().propose(node)

    def propose_for_real_support(
        self, node_var: Variable
    ) -> Tuple[bool, Tensor, Tensor]:
        """
        Proposes a new value for the node by drawing a sample from the proposal
        distribution (MultivariateNormal) and compute the log probability of
        the new draw coming from this proposal distribution log(P(X->X')).

        :param node: the node for which we'll need to propose a new value for.
        :param node_var: the Variable associated with the node
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        node_val = node_var.value
        number_of_variables = len(self.world_.variables_) - len(
            self.world_.observations_
        )
        proposal_distribution = node_var.proposal_distribution
        if proposal_distribution is None or number_of_variables != 1:
            is_valid, mean, covariance = self.compute_normal_mean_covar(node_var)
            if not is_valid:
                return False, tensor(0.0), tensor(0.0)
            proposal_distribution = dist.MultivariateNormal(mean, covariance)
            node_var.proposal_distribution = proposal_distribution
        # pyre-fixme
        new_value = proposal_distribution.sample().reshape(node_val.shape)
        new_value.requires_grad_(True)
        negative_proposal_log_update = (
            -1
            # pyre-fixme
            * proposal_distribution.log_prob(new_value).sum()
        )
        return True, new_value, negative_proposal_log_update

    def propose_for_simplex_support(
        self, node_var: Variable
    ) -> Tuple[bool, Tensor, Tensor]:
        """
        Proposes a new value for the node by drawing a sample from the proposal
        distribution (Dirichlet) and compute the log probability of
        the new draw coming from this proposal distribution log(P(X->X')).

        :param node: the node for which we'll need to propose a new value for.
        :param node_var: the Variable associated with the node
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        node_val = node_var.value
        number_of_variables = len(self.world_.variables_) - len(
            self.world_.observations_
        )
        proposal_distribution = node_var.proposal_distribution
        if proposal_distribution is None or number_of_variables != 1:
            is_valid, alpha = self.compute_alpha(node_var)
            if not is_valid:
                return False, tensor(0.0), tensor(0.0)
            proposal_distribution = dist.Dirichlet(alpha)
            node_var.proposal_distribution = proposal_distribution
        # pyre-fixme
        new_sample = proposal_distribution.sample()
        negative_proposal_log_update = (
            -1
            # pyre-fixme
            * proposal_distribution.log_prob(new_sample).sum()
        )
        if isinstance(node_var.distribution, dist.Beta):
            new_value = new_sample.transpose(-1, 0)[0].T.reshape(node_val.shape)
        else:
            new_value = new_sample.reshape(node_val.shape)
        return True, new_value, negative_proposal_log_update

    def propose_for_hspace_support(
        self, node_var: Variable
    ) -> Tuple[bool, Tensor, Tensor]:
        """
        Proposes a new value for the node by drawing a sample from the proposal
        distribution (Gamma) and compute the log probability of
        the new draw coming from this proposal distribution log(P(X->X')).

        :param node: the node for which we'll need to propose a new value for.
        :param node_var: the Variable associated with the node
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        node_val = node_var.value
        number_of_variables = len(self.world_.variables_) - len(
            self.world_.observations_
        )
        proposal_distribution = node_var.proposal_distribution
        if proposal_distribution is None or number_of_variables != 1:
            is_valid, alpha, beta = self.compute_alpha_beta(node_var)
            if not is_valid:
                return False, tensor(0.0), tensor(0.0)

            proposal_distribution = dist.Gamma(alpha, beta)
            node_var.proposal_distribution = proposal_distribution

        # pyre-fixme
        new_value = proposal_distribution.sample().reshape(node_val.shape)
        new_value.requires_grad_(True)
        negative_proposal_log_update = (
            -1
            # pyre-fixme
            * proposal_distribution.log_prob(new_value).sum()
        )
        return True, new_value, negative_proposal_log_update
