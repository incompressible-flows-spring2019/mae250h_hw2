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

# Test that subtraction of node data works
p2 = deepcopy(p)
@test LinearAlgebra.norm(p2 - p) == 0.0

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

# The rot of constant data should be uniformly 0
@test norm(w) == 0.0

# Should not be able to accept other types of data
@test_throws MethodError HW2.rot(w)

end

@testset "Gradient" begin

nx = 50; ny = 25
p = HW2.CellData(nx,ny)
fill!(p.data,1.0)
q = HW2.gradient(p)

@test typeof(q) <: HW2.EdgeData

# The gradient of constant data should be uniformly 0
@test norm(q) == 0.0

# Should not be able to accept other types of data
@test_throws MethodError HW2.gradient(q)

end

@testset "Curl" begin

nx = 50; ny = 25
s = HW2.NodeData(nx,ny)
fill!(s.data,1.0)
q = HW2.curl(s)

@test typeof(q) <: HW2.EdgeData

# The curl of constant data should be uniformly 0
@test norm(q) == 0.0

# Should not be able to accept other types of data
@test_throws MethodError HW2.curl(HW2.CellData(s))

end

@testset "Laplacian" begin

# test that Laplacian of cell data is equivalent to divergence of gradient
nx = 50; ny = 25
p = HW2.CellData(nx,ny)
p.data .= randn(size(p.data))
lapp = HW2.laplacian(p)
lapp2 = HW2.divergence(HW2.gradient(p))
err = HW2.CellData(lapp)
err.data .= lapp.data - lapp2.data
@test norm(err) < 1e-13

end

@testset "Nullspaces" begin

nx = 50; ny = 25

# set up some random node data
s = HW2.NodeData(nx,ny)
s.data .= randn(size(s.data))

# take the divergence of the curl
p = HW2.divergence(HW2.curl(s))

# Result should be very small
@test norm(p) < 1e-14

# do the same test with rot of the gradient
p = HW2.CellData(nx,ny)
p.data .= randn(size(p.data))

s = HW2.rot(HW2.gradient(p))
@test norm(s) < 1e-14

end

@testset "Translations" begin

# Translate edge data to edge data (x -> y, y -> x)
nx = 5; ny = 7
q1 = HW2.EdgeData(nx,ny)
fill!(q1.qx,1.0)
q2 = HW2.EdgeData(q1)
HW2.translate!(q2,q1)

# The new data should be ones on all non-ghost y edges
@test dot(q2,q2) == 1.0

end
