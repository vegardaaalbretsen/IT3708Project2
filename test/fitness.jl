using Test
using HomeCareGA

const SPLIT = -1

@info ("Testing fitness evaluation")
@testset verbose=true "Fitness" begin
    # Small instance: depot=0, patients=1..3
    # travel matrix (4x4): nodes [0,1,2,3]
    T = [
        0.0  5.0  5.0  5.0;
        5.0  0.0  1.0  1.0;
        5.0  1.0  0.0  1.0;
        5.0  1.0  1.0  0.0
    ]

    inst = HCInstance(
        name="toy",
        N=3,
        travel=T,
        start_time=[0.0, 0.0, 0.0],
        end_time=[100.0, 100.0, 100.0],
        care_time=[10.0, 10.0, 10.0],
        demand=[1.0, 1.0, 1.0],
        capacity_nurse=2.0,
        depot_return_time=40.0
    )

    weights = FitnessWeights(
        w_travel=1.0,
        w_wait=0.0,
        w_capacity=1000.0,
        w_return=1000.0,
        w_late=1000.0,
        w_early=0.0
    )

    schedule_lo = PenaltySchedule(min_scale=1.0, max_scale=1.0, power=1.0, mag_scale=0.0)
    schedule_hi = PenaltySchedule(min_scale=1.0, max_scale=10.0, power=1.0, mag_scale=0.0)

    @testset "No violations => penalties zero" begin
        # two routes: [1,2] and [3]
        chrom = [1,2,SPLIT,3]

        b = fitness_breakdown(chrom, inst; weights=weights, schedule=schedule_lo, generation=1, max_generations=100)

        @test b.cap_violation == 0.0
        @test b.late == 0.0
        @test b.return_overtime == 0.0
        @test b.total ≈ (weights.w_travel*b.travel + weights.w_wait*b.wait)
    end

    @testset "Capacity violation is penalized" begin
        # single route visits all 3 -> demand=3 exceeds cap=2 by 1
        chrom = [1,2,3]
        b = fitness_breakdown(chrom, inst; weights=weights, schedule=schedule_lo)
        @test b.cap_violation > 0.0
        @test b.total > b.travel  # should include penalty
    end

    @testset "Return-time overtime is penalized" begin
        # make return time very tight
        inst2 = HCInstance(inst; depot_return_time=10.0)
        chrom = [1,2]
        b = fitness_breakdown(chrom, inst2; weights=weights, schedule=schedule_lo)
        @test b.return_overtime > 0.0
    end

    @testset "Late arrival is penalized" begin
        # Set a tight time window for patient 2
        inst3 = HCInstance(inst; start_time=[0.0, 0.0, 0.0], end_time=[100.0, 1.0, 100.0])
        chrom = [1,2]  # travel to 1 (5) + care(10) then to 2 (1) => arrive 16 > 1 => late
        b = fitness_breakdown(chrom, inst3; weights=weights, schedule=schedule_lo)
        @test b.late > 0.0
    end

    @testset "Dynamic ramp increases penalty over generations" begin
        chrom = [1,2,3]  # capacity violation

        f1 = fitness(chrom, inst; weights=weights, schedule=schedule_hi, generation=1, max_generations=100)
        f2 = fitness(chrom, inst; weights=weights, schedule=schedule_hi, generation=100, max_generations=100)

        @test f2 > f1
    end
end