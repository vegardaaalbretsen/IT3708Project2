module HomeCareGA

using JSON3
using Plots
using Printf
using Random

include("model.jl")
include("instance_io.jl")
include("evaluation.jl")
include("repair.jl")
include("operators.jl")
include("ga.jl")
include("reporting.jl")
include("plotting.jl")
include("config.jl")
include("api.jl")

export GAConfig
export default_config
export load_instance
export solve_instance

end
