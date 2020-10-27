import torch
import torch.optim
import torch.distributions as dist

from .IAF import FlowStack


class VariationalApproximation(dist.distribution.Distribution):
    def __init__(self, target):
        super(VariationalApproximation, self).__init__()
        self.p = target
        self.d = 2
        self.flow_stack = FlowStack(dim=2, n_flows=8)

    def train(self, epochs=100, lr=1e-2):
        sample_shape = (100, 2)
        optim = torch.optim.Adam(self.flow_stack.parameters(), lr=lr)

        for i in range(epochs):
            z0, zk, mu, log_var, ldj = self.flow_stack(shape=sample_shape)

            # negative ELBO loss

            # entropy H(Q)
            loss = (
                dist.Normal(mu, torch.exp(0.5 * log_var)).log_prob(z0).sum()
            )  # Q(z_0)
            loss -= ldj.sum()  # transport to Q(z_k) via - sum log det jac

            # negative cross-entropy -H(Q,P)
            loss -= self.p.log_prob(zk).sum()

            # normalize by batch size
            loss /= z0.size(0)

            loss.backward()
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                print(loss.item())

    def sample(self, sample_shape=torch.Size()):
        _, xs, _, _, _ = self.flow_stack(shape=sample_shape)
        return xs.detach()

    def log_prob(self, value):
        # if z' = f(z), Q(z') = Q(z) |det df/dz|^{-1}
        z = value
        ldj = 0.0
        for flow in reversed(self.flow_stack.flow):
            h = torch.zeros(flow.context_size)
            s_t = flow.s_t(z, h) + 1.5
            sigma_t = torch.sigmoid(s_t)
            m_t = flow.m_t(z, h)
            ldj += flow.log_det_jac(sigma_t)
            z_prev = (z - (1 - sigma_t) * m_t) / sigma_t
            z = z_prev
        return (
            dist.Independent(
                dist.Normal(self.flow_stack.mu, torch.exp(self.flow_stack.log_var / 2)),
                1,
            ).log_prob(z)
            - ldj
        )
