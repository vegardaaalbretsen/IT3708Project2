using Test
using HomeCareGA.Fitness: HCInstance
using HomeCareGA: optimal_split

@info("Testing Split operators")

@testset verbose=true "optimal_split Test" begin

    # Helper: validate structure of split result
    function _validate_solution(result, perm)
        @test result !== nothing
        @test sort(filter(!=(-1), result)) == perm
    end


    @testset "Capacity forces split into two routes" begin
        travel = [
            0.0 1 1 1;
            1 0 1 1;
            1 1 0 1;
            1 1 1 0
        ]

        inst = HCInstance(
            name = "cap_test",
            N = 3,
            travel = travel,
            start_time = [0.0, 0.0, 0.0],
            end_time   = [100.0, 100.0, 100.0],
            care_time  = [0.0, 0.0, 0.0],
            demand     = [1.0, 1.0, 1.0],
            capacity_nurse = 2.0,
            depot_return_time = 100.0
        )

        perm = [1, 2, 3]
        result, cost = optimal_split(inst, perm)

        _validate_solution(result, perm)

        # Must have exactly one split
        @test count(==(-1), result) == 1

        # Cost must be optimal (5)
        @test cost ≈ 5.0
    end


    @testset "Time window forces split" begin
        travel = [
            0.0 1 1;
            1 0 1;
            1 1 0
        ]

        inst = HCInstance(
            name = "tw_test",
            N = 2,
            travel = travel,
            start_time = [0.0, 0.0],
            end_time   = [100.0, 2.5],  # tight window for patient 2
            care_time  = [1.0, 0.0],
            demand     = [1.0, 1.0],
            capacity_nurse = 10.0,
            depot_return_time = 100.0
        )

        perm = [1, 2]
        result, cost = optimal_split(inst, perm)

        _validate_solution(result, perm)

        # Must split into two routes
        @test count(==(-1), result) == 1
        @test cost > 0
    end


    @testset "Infeasible instance returns nothing, Inf" begin
        travel = [
            0.0 10;
            10 0
        ]

        inst = HCInstance(
            name = "infeasible_test",
            N = 1,
            travel = travel,
            start_time = [0.0],
            end_time   = [5.0],
            care_time  = [0.0],
            demand     = [1.0],
            capacity_nurse = 10.0,
            depot_return_time = 5.0
        )

        perm = [1]
        result, cost = optimal_split(inst, perm)

        @test result === nothing
        @test cost == Inf
    end


    @testset "Existing -1 delimiters are ignored" begin
        travel = [
            0.0 1 1;
            1 0 1;
            1 1 0
        ]

        inst = HCInstance(
            name = "delimiter_test",
            N = 2,
            travel = travel,
            start_time = [0.0, 0.0],
            end_time   = [100.0, 100.0],
            care_time  = [0.0, 0.0],
            demand     = [1.0, 1.0],
            capacity_nurse = 10.0,
            depot_return_time = 100.0
        )

        perm_with_delims = [1, -1, 2]
        result, cost = optimal_split(inst, perm_with_delims)

        _validate_solution(result, [1,2])

        # Should produce single route
        @test count(==(-1), result) == 0
        @test cost ≈ 3.0
    end


    @testset "Single route when fully feasible" begin
        travel = [
            0.0 2 2;
            2 0 1;
            2 1 0
        ]

        inst = HCInstance(
            name = "single_route_test",
            N = 2,
            travel = travel,
            start_time = [0.0, 0.0],
            end_time   = [100.0, 100.0],
            care_time  = [0.0, 0.0],
            demand     = [1.0, 1.0],
            capacity_nurse = 10.0,
            depot_return_time = 100.0
        )

        perm = [1, 2]
        result, cost = optimal_split(inst, perm)

        _validate_solution(result, perm)

        @test count(==(-1), result) == 0
        @test cost ≈ 5.0
    end

end