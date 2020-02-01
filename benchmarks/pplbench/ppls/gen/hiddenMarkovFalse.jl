
using Gen

# Define the statistical model as a generative function.
# The inputs are the prior, and the output is the observed variables.
# Latent variables can be output as well (under some conditions).
@gen function model(k, n, concentration, mu_loc, mu_scale, sigma_shape, sigma_rate)

    # Define the model of thetas, mus, sigmas
    mus = Float64[]
    sigmas = Float64[]
    # Gen does not support Dirichlet distributions ... hard code a value for theta
    thetas = [[.1,.3,.6],[.1,.3,.6],[.1,.3,.6]]

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
    return ys
end

# Helper function for inference: single-site
function do_update(tr, k, n)
    for i in 1:k
        (tr, _) = mh(tr, select(:params => i => :mu))
        (tr, _) = mh(tr, select(:params => i => :sigma))
    end
    for i in 1:n
        (tr, _) = mh(tr, select(:data => i => :x))
    end
    return tr
end

# Helper function for inference: blocked
function do_block_update(tr, n)
    for i in 1:n
        (tr, _) = mh(tr, select(:data => i => :x,
                            :params => (:data => i => :x) => :mu,
                            :params => (:data => i => :x) => :sigma))
    end
    return tr
end

function main(m, n, k, concentration, mu_loc, mu_scale, sigma_shape, sigma_rate, num_samples, ys)
    # Define observations using dict-like structure
    observations = Gen.choicemap()
    for (i,y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # Setup inference
    xs = []

    mus = []
    sigmas = []
    # The second argument provides the arguments to m, which is a Generative function.
    (tr, _) = generate(m, (k, n, concentration, mu_loc, mu_scale, sigma_shape, sigma_rate), observations)
    # Run MH imperatively, using a for-loop. Meanwhile, record scores (log-probs).
    for iter in 1:num_samples
        tr = do_update(tr, k, n)
        #tr = do_block_update(tr, n)

        # Store results
        for i in 2:n
            push!(xs, tr[:data => i => :x])
        end
        for i in 1:k
            push!(mus, tr[:params => i => :mu])
            push!(sigmas, tr[:params => i => :sigma])
        end
    end
    return [xs, mus, sigmas]
end

# scores = do_inference(model, ys)
#savefig(plot(1:length(scores), scores), "fn.pdf")

#result = main(model, 20,3,0.8,1.0,5.0,3.0,3.0,50,[-7.5107675, 0.5754061, 1.9674139, 2.0757697, 0.16821182, 1.7968048, 2.5447257, 3.4935503, 1.3484366, 1.9544795, 5.3395147, -4.3594966, 1.9002563, -0.716866, 0.88932806, 0.85559213, -7.069362, -6.554332, -3.9915106, 0.05220163])
#print(result)
