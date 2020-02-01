# Load libraries.
using Turing, Plots, Random

# Turn on progress monitor.
Turing.turnprogress(true);

# Set a random seed and use the forward_diff AD mode.

function main(N, K, concentration, mu_loc, mu_scale, sigma_shape, sigma_scale, num_samples, observations)
    # Define the emission variable.
    # y = observations;

    # Plot the data we just made.
    # plot(y, xlim = (0,15), ylim = (-1,5), size = (500, 250))

    # Turing model definition.
    @model BayesHmm(y, K, N, mu_loc, mu_scale, sigma_shape, sigma_scale) = begin
        # State sequence.
        x = tzeros(Int, N)

        # Emission distribution.
        mu = Vector(undef, K)
        sigma = Vector(undef, K)

        # Transition matrix.
        T = Vector{Vector}(undef, K)

        # Assign distributions to each element
        # of the transition matrix and the
        # emission distribution.
        for i = 1:K
            T[i] ~ Dirichlet(ones(K) * concentration / K)
            mu[i] ~ Normal(mu_loc, mu_scale)
            sigma[i] ~ Gamma(sigma_shape, sigma_scale)
        end

        # Observe each point of the input.
        x[1] ~ Categorical(K)
        y[1] ~ Normal(mu[x[1]], sigma[x[1]])

        for i = 2:N
            x[i] ~ Categorical(vec(T[x[i-1]]))
            y[i] ~ Normal(mu[x[i]], sigma[x[i]])
        end
    end


    g = Gibbs(HMC(0.001, 7, :mu, :sigma, :T), PG(20, :x))
    chain = sample(BayesHmm(observations, K, N, mu_loc, mu_scale, sigma_shape, sigma_scale), g, num_samples);

    # println(chain)
    # Return values of sampled chain
    [get(chain, :T), get(chain, :mu), get(chain, :sigma), get(chain, :x)]
end
