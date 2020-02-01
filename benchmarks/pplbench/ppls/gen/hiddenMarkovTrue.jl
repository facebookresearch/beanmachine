
using Gen

# Define the statistical model as a generative function.
# The inputs are the prior, and the output is the observed variables.
# Latent variables can be output as well (under some conditions).
@gen function model(k, n, concentration, mu_loc, mu_scale, sigma_shape, sigma_rate, thetas, mus, sigmas)

    # Define the model of thetas, mus, sigmas
    mus = Float64[]
    sigmas = Float64[]

    for i in 1:k
        mu = @trace(normal(mu_loc, mu_scale), :params => i => :mu)
        push!(mus,mu)
        sigma = @trace(gamma(sigma_shape, sigma_rate), :params => i => :sigma)
        push!(sigmas,sigma)
        #theta = @trace(dirichlet(ones(k) * concentration / k), :params => i => :theta)
        #push!(thetas,theta)
    end

    # Define the hidden states and observations
    xs = Float64[]
    ys = Float64[]

    x = 1
    push!(xs, x)
    y = @trace(normal(mus[x], sigmas[x]), :data => 1 => :y )
    push!(ys, y)

    for i in 2:n
        x = @trace(categorical(thetas[x]), :data => i => :x )
        push!(xs, x)
        y = @trace(normal(mus[x], sigmas[x]), :data => i => :y )
        push!(ys, y)
    end
    return ys, mus, sigmas
end

# Helper function for inference: single-site
function do_ss_update(tr, k, n)
    #for i in 1:k
    #    (tr, _) = mh(tr, select(:params => i => :mu))
    #    (tr, _) = mh(tr, select(:params => i => :sigma))
    #end
    for i in 1:n
        (tr, _) = mh(tr, select(:data => i => :x))
    end
    return tr
end

# Helper function for inference: global
function do_global_update(tr, k, n)
    selection = select()
    #for i in 1:k
    #    push!(selection,(:params => i => :mu))
    #    push!(selection,(:params => i => :sigma))
    #end
    for i in 1:n
        push!(selection,(:data => i => :x))
    end
    (tr, _) = mh(tr, selection)
    return tr
end

# Helper function for inference: blocked
# Not that k is un-used, it is only satisfying the typical interface
function do_block_update(tr, k, n)
    for i in 1:n
        (tr, _) = mh(tr, select(:data => i => :x,
                            :params => (:data => i => :x) => :mu,
                            :params => (:data => i => :x) => :sigma))
    end
    return tr
end

function main(m, n, k, concentration, mu_loc, mu_scale, sigma_shape, sigma_rate, num_samples, ys, thetas, mus, sigmas)
    # Define observations using dict-like structure
    observations = Gen.choicemap()
    for (i,y) in enumerate(ys)
        observations[(:y, i)] = y
    end
    for (i,mu) in enumerate(mus)
        observations[(:mu, i)] = mu
    end
    for (i,sigma) in enumerate(sigmas)
        observations[(:sigma, i)] = sigma
    end

    # Setup inference
    xs = [1]

    # The second argument provides the arguments to m, which is a Generative function.
    (tr, _) = generate(m, (k, n, concentration, mu_loc, mu_scale, sigma_shape, sigma_rate, thetas, mus, sigmas), observations)
    # Run MH imperatively, using a for-loop. Meanwhile, record scores (log-probs).
    for iter in 1:(num_samples-1)
        tr = do_ss_update(tr, k, n)

        # Store results
        #for i in 2:n
        #    push!(xs, tr[:data => n => :x])
        #end
        push!(xs, tr[:data => n => :x])
    end
    return xs
end

# scores = do_inference(model, ys)
#savefig(plot(1:length(scores), scores), "fn.pdf")
