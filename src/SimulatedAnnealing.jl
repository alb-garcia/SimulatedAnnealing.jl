module SimulatedAnnealing

using Distributed
using Statistics

export SA, run, prun, geometric_schedule_fn, constant_Ebsf_done_fn, constant_E_done_fn,
    n_steps_done_fn, n_moves, n_accepted, n_max_moves!, n_max_moves, n_max_accepted,
    n_max_accepted!, set_current_solution!, current_solution, mean_E_total, mean_E_accepted,
    var_E_total, var_E_accepted, Ebsf, bsf, E, T


mutable struct SA{ET<:Real , ST<:Any}
    s::Vector{ST}     # current solution history
    s_best::Vector{ST} # best solution so far history
    T::Vector{AbstractFloat}      # Temperature history
    E::Vector{ET}      # Energy of current solution history
    Ebsf::Vector{ET} # Energy of best solution so far history
    n_max_moves::Int  # maximum number of moves before new T step
    n_max_accepted::Int # maximum number of accepted solutions before new T step
    step::Int         # current step number
    n_moves::Vector{Int}      # number of moves attempted per step
    n_accepted::Vector{Int}   # max. number of accepted states per step
    mean_E_total::Vector{Float64}  # average E of all tried neighbor candidates
    mean_E_accepted::Vector{Float64} # average E of all accepted neighbors
    var_E_total::Vector{Float64}  # average E of all tried neighbor candidates
    var_E_accepted::Vector{Float64} # average E of all accepted neighbors
    schedule::Function
    done::Function
    guess_T0::Bool
    p0::Float64    
    fused_neighbor_energy::Bool
    energy_statistics::Bool
    solution_history::Bool
    report_every::Int
    current_solution::ST
    bsf::ST
    
    function SA{ET,ST}(;schedule, done, n_max_moves, n_max_accepted,
                      guess_T0 = true,
                      p0 = 0.5,
                      fused_neighbor_energy = false,
                      energy_statistics = true,
                      solution_history = false,
                      report_every = 0) where ET<:Real where ST

        new{ET,ST}(ST[], #s
                  ST[],   #s_best
                  Vector{Float64}(),    #T
                  Vector{ET}(),    #E
                  Vector{ET}(),    #Ebsf
                  n_max_moves,        #n_max_moves
                  n_max_accepted,     #n_max_accepted
                  0, #step
                  Int[], # n_moves
                  Int[],  #n_accepted
                  Vector{Float64}(), #mean_E_total
                  Vector{Float64}(), #mean_E_accepted
                  Vector{Float64}(), #var_E_total
                  Vector{Float64}(), #var_E_accepted
                  schedule,
                  done,
                  guess_T0,
                  p0,
                  fused_neighbor_energy,
                  energy_statistics,
                  solution_history,
                  report_every)
    end
end

""" returns total number of moves trialed in last annealing step"""
n_moves(sa::SA{ET,ST} where ET where ST) = sa.n_moves[end]

""" returns number of accepted moves in last annealing step"""
n_accepted(sa::SA{ET,ST} where ET where ST) = sa.n_accepted[end]

""" set max number of trialed moves in an annealing step"""
n_max_moves!(sa::SA{ET,ST} where ET where ST,n) = sa.n_max_moves = n

""" returns max number of trialed moves in an annealing step"""
n_max_moves(sa::SA{ET,ST} where ET where ST) = sa.n_max_moves

""" returns max number of accepted moves in an annealing step"""
n_max_accepted(sa::SA{ET,ST} where ET where ST) = sa.n_max_accepted

""" set max number of accepted moves in an annealing step"""
n_max_accepted!(sa::SA{ET,ST} where ET where ST,n) = sa.n_max_accepted = n

""" set the current solution"""
set_current_solution!(sa::SA{ET,ST} where ET where ST, solution) = sa.current_solution = solution

""" returns the current candidate solution"""
current_solution(sa::SA{ET,ST} where ET where ST) = sa.current_solution

""" returns the mean energy of all trialed moves in last annealing step"""
mean_E_total(sa::SA{ET,ST} where ET where ST) = sa.mean_E_total[end]

""" returns the mean energy of all accepted moves in last annealing step"""
mean_E_accepted(sa::SA{ET,ST} where ET where ST) = sa.mean_E_accepted[end]

""" returns the energy variance of all trialed moves in last annealing step"""
var_E_total(sa::SA{ET,ST} where ET where ST) = sa.var_E_total[end]

""" returns the energy variance of all accepted moves in last annealing step"""
var_E_accepted(sa::SA{ET,ST} where ET where ST) = sa.var_E_accepted[end]

""" returns the energy of the Best So Far solution over the whole annealing process"""
Ebsf(sa::SA{ET,ST} where ET where ST) = sa.Ebsf[end]

""" returns the Best So Far solution over the whole annealing process"""
bsf(sa::SA{ET,ST} where ET where ST) = sa.bsf

""" returns the final energy of the previous annealing step"""
E(sa::SA{ET,ST} where ET where ST) = sa.E[end]

""" returns the temperature of the previous annealing step"""
T(sa::SA{ET,ST} where ET where ST) = sa.T[end]

""" returns the history (vector of length = number of annealing steps so far) of 
some quantity in annealing process sa. Symbol s determines the quantity:
:n_moves     - number of moves trialed per step
:n_accepted  - number of moves accepted per step
:mean_E_total - average energy of all moves trialed per step
:mean_E_accepted - average energy of all moves accepted per step
:var_E_total     - energy variance of all moves trialed per step
:var_E_accepted  - energy variance of all moves accepted per step
:Ebsf            - energy of Best solution So Far per step
:E               - final energy per annealing step
:T               - temperature per annealing step
"""
history(sa::SA{ET,ST} where ET where ST, s::Symbol) = getfield(sa, s)

function history(sa::SA{ET,ST} where ET where ST, s::Symbol, n::Int)
    v = getfield(sa,s)
    N = length(v)
    if n > 0
        @view v[n:N]
    else
        @view v[N + n : N]
    end
end




geometric_schedule_fn(alpha) = sa::SA -> sa.T[end] * alpha

function constant_E_done_fn(n_steps)
    function (sa::SA)
        if sa.step < n_steps
            false
        else
            N = sa.step
            E_now = sa.E[N]
            all(x->x ≈ E_now, sa.E[N-(n_steps-1):N-1])
        end
    end
end

function constant_Ebsf_done_fn(n_steps)
    function (sa::SA)
        if sa.step < n_steps
            false
        else
            N = sa.step
            E_now = sa.Ebsf[N]
            all(x->x ≈ E_now, sa.Ebsf[N-(n_steps-1):N-1])
        end
    end
end

function n_steps_done_fn(n_steps)
    ndone(sa::SA) = sa.step >= n_steps
end

function arithmetic_schedule_fn(a,b)
    function asched(sa::SA)
        a*sa.T[end] + b
    end
end

function initial_temperature(s0, energy::Function, neighbor::Function, p0)
    ΔE_avg = 0.0
    E = energy(s0)
    for i in 1:1000
        ΔE_avg += abs(E - (energy ∘ neighbor)(copy(s0)))
    end
    ΔE_avg = ΔE_avg / 1000
    
    return -(ΔE_avg / log(p0))
end

function report(sa::SA)
    println("- step $(sa.step), T = $(sa.T[end]), E = $(sa.E[end]), Ebsf = $(sa.Ebsf[end])")
end

function prun(;sas::Vector{SA{I,J}} where I where J,
              s0s,
              neighbor::Function,
              energy::Function,
              nprocs::Int,
              T0::Float64 = -1.0,
              revert::Union{Function, Nothing} = nothing)

    #awkward indexed loop, but @threads doesn't support i,j iteration over zips
    Threads.@threads for i in 1:length(sas)
        run(sa = sas[i], s0 = s0s[i], neighbor = neighbor, energy = energy,
            T0 = T0, revert = revert)
    end
    return sas

end


function run(;sa::SA{ET,ST} where ET where ST,
             s0,
             neighbor::Function,
             energy::Function,
             T0::Float64 = -1.0,
             revert::Union{Function, Nothing} = nothing)

    
    # either guess initial temperature, or user-given T0
    # error-out if T0 not given and no guess
    if sa.guess_T0
        neighbor_fn = sa.fused_neighbor_energy ? x -> neighbor(x)[1] : neighbor
        T = initial_temperature(s0, energy, neighbor_fn, sa.p0)
    elseif T0 < 0.0
        error("if guess_initial_temperature = false, a T0 > 0.0 must be set")
    else
        T = T0
    end
    
    # initialize best, current solutions, energies and number of moves
    E = energy(s0)
    zero_E = zero(E)
    s = s0
    sa.step = 0
    s_best = s
    Ebsf = E
    
    # allocate energy vectors once and reuse for every annealing step
    if sa.energy_statistics
        E_candidates = zeros(typeof(E), sa.n_max_moves)
        E_accepted  = zeros(typeof(E), sa.n_max_accepted)
    end

    # initialize copy of current solution
    scopy = copy(s0)

    while !sa.done(sa)
        n_accepted = 0
        n_moves = 0

        for i in 1:sa.n_max_moves

            if sa.fused_neighbor_energy
                trial,ΔE = neighbor(scopy)
                E_trial = E + ΔE
            else
                trial = neighbor(scopy)
                E_trial = energy(trial)
                ΔE = E_trial - E
            end
            
            n_moves += 1

            if sa.energy_statistics
                E_candidates[n_moves] = E_trial
            end

            # accept a neighbor with lower energy or with P = e^(-ΔE/T)
            if ΔE <= zero_E ||  @fastmath rand() < exp(-ΔE / T)
                s = trial
                E = E_trial
                n_accepted += 1
                
                if sa.energy_statistics
                    E_accepted[n_accepted] = E_trial
                end

                scopy = copy(s)
            else
                scopy = revert != nothing ? revert(scopy, s) : copy(s)
            end

            # update best solution/energy
            if E_trial < Ebsf
                Ebsf = E_trial
                s_best = trial
            end

            # max number of accepted solutions reached? -> next annealing step
            if n_accepted == sa.n_max_accepted
                break
            end
        end

        if sa.energy_statistics
            push!(sa.mean_E_total, Statistics.mean(E_candidates[1:n_moves]))
            push!(sa.mean_E_accepted, Statistics.mean(E_accepted[1:n_accepted]))
            push!(sa.var_E_total, Statistics.var(E_candidates[1:n_moves]))
            push!(sa.var_E_accepted, Statistics.var(E_accepted[1:n_accepted]))
        end

        if sa.solution_history
            push!(sa.s, s)
            push!(sa.s_best, s_best)
        end
        
        push!(sa.n_moves, n_moves)
        push!(sa.n_accepted, n_accepted)
        push!(sa.E, E)
        push!(sa.Ebsf, Ebsf)
        

        sa.step += 1
        push!(sa.T,T)
        sa.current_solution = s
        sa.bsf = s_best

        if sa.report_every > 0 && sa.step % sa.report_every == 0
            report(sa)
        end
        
        T = sa.schedule(sa)
    end

    if sa.report_every >= 0
        report(sa)
    end
    
    return sa
end


end
