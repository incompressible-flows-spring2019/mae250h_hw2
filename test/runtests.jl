# A package of useful testing commands
using Test

# We need to tell this test to load the module we are testing
using HW2

# The Random package is useful for making tests as arbitrary as possible
using Random

# The LinearAlgebra package is useful for some norms
using LinearAlgebra

@testset "Cell centered data" begin

nx = 50; ny = 25
p = HW2.CellData(rand(nx+2,ny+2))

@test size(p) == (nx+2,ny+2)

@test typeof(p) <: HW2.GridData{nx,ny}

p = HW2.CellData(nx,ny)
# choose a random interior point to set equal to 1
i, j = rand(2:nx+1), rand(2:ny+1)
p.data[i,j] = 1.0

@test LinearAlgebra.norm(p)*sqrt(nx*ny) == 1.0

# Test that subtraction of cell data works
p2 = deepcopy(p)
@test LinearAlgebra.norm(p2 - p) == 0.0

end

@testset "Edge data" begin

nx = 50; ny = 25
p = HW2.CellData(rand(nx+2,ny+2))
q = HW2.EdgeData(p)

@test size(q.qx) == (nx+1,ny+2)
@test size(q.qy) == (nx+2,ny+1)

@test typeof(q) <: HW2.GridData{nx,ny}

# Test that subtraction of edge data works
q2 = deepcopy(q)
@test LinearAlgebra.norm(q2 - q) == 0.0

end

@testset "Node data" begin

nx = 50; ny = 25
p = HW2.NodeData(nx,ny)

@test size(p) == (nx+1,ny+1)

@test typeof(p) <: HW2.GridData{nx,ny}

end

@testset "Divergence" begin

nx = 50; ny = 25
q = HW2.EdgeData(nx,ny)
# choose a random edge in interior of grid and set it to 1
i, j = rand(2:nx), rand(2:ny+1)
q.qx[i,j] = 1

p = HW2.divergence(q)
# Result should be of type CellData
@test typeof(p) <: HW2.CellData

# Should not be able to accept other types of data
@test_throws MethodError HW2.divergence(p)

end

@testset "Rot" begin

nx = 50; ny = 25
q = HW2.EdgeData(nx,ny)
fill!(q.qx,1.0)
w = HW2.rot(q)

@test typeof(w) <: HW2.NodeData

# Should not be able to accept other types of data
@test_throws MethodError HW2.rot(w)

end
