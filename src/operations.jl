#=
Here we will set up operations to be carried out on our grid data. Note that
all of these operations are to be carried out in index space.
divergence
grad
curl
rot
Laplacian
translation (e.g. translating from cell data to edge data)
dot products for edge data and node data
=#

# The LinearAlgebra package contains some useful functions on arrays
# We import some of its operations to extend them to our data here
using LinearAlgebra
import LinearAlgebra:dot, norm

#============  INNER PRODUCTS AND NORMS ===============#

# To compute inner products, we will extend the Julia function `dot`. Note that
# we exclude the ghost cells from the dot product.
"""
    dot(p1::CellData,p2::CellData) -> Real

Computes the inner product between two sets of cell-centered data on the same grid.
"""
function dot(p1::CellData{NX,NY},p2::CellData{NX,NY}) where {NX,NY}
  return dot(p1.data[2:NX+1,2:NY+1],p2.data[2:NX+1,2:NY+1])/(NX*NY)
end

"""
    dot(p1::EdgeData,p2::EdgeData) -> Real

Computes the inner product between two sets of edge data on the same grid.
"""
function dot(p1::EdgeData{NX,NY},p2::EdgeData{NX,NY}) where {NX,NY}

  # interior
  tmp = dot(p1.qx[2:NX,2:NY+1],p2.qx[2:NX,2:NY+1]) +
        dot(p1.qy[2:NX+1,2:NY],p2.qy[2:NX+1,2:NY])

  # boundaries
  tmp += 0.5*dot(p1.qx[1,2:NY+1],   p2.qx[1,2:NY+1])
  tmp += 0.5*dot(p1.qx[NX+1,2:NY+1],p2.qx[NX+1,2:NY+1])
  tmp += 0.5*dot(p1.qy[2:NX+1,1],   p2.qy[2:NX+1,1])
  tmp += 0.5*dot(p1.qy[2:NX+1,NY+1],p2.qy[2:NX+1,NY+1])

  return tmp/(NX*NY)
end

# DEVELOP A NORM FOR NODE DATA


"""
    norm(p::GridData) -> Real

Computes the L2 norm of data on a grid.
"""
function norm(p::GridData{NX,NY}) where {NX,NY}
  return sqrt(dot(p,p))
end

# This function computes an integral by just taking the inner product with
# another set of cell data uniformly equal to 1
"""
    integrate(p::CellData) -> Real

Computes a numerical quadrature of the cell-centered data.
"""
function integrate(p::CellData{NX,NY}) where {NX,NY}
  p2 = CellData(p)
  fill!(p2.data,1) # fill it with ones
  return dot(p,p2)
end

"""
    integrate(p::NodeData) -> Real

Computes a numerical quadrature of the node data.
"""
function integrate(p::NodeData{NX,NY}) where {NX,NY}
  p2 = NodeData(p)
  fill!(p2.data,1) # fill it with ones
  return dot(p,p2)
end

#=============== DIFFERENCING OPERATIONS ==================#

"""
    divergence(q::EdgeData) -> CellData

Compute the discrete divergence of edge data `q`, returning cell-centered
data on the same grid.
"""
function divergence(q::EdgeData{NX,NY}) where {NX,NY}
   p = CellData(q)
   # Loop over interior cells
   for j in 2:NY+1, i in 2:NX+1
     p.data[i,j] = q.qx[i,j] - q.qx[i-1,j] + q.qy[i,j] - q.qy[i,j-1]
   end
   return p

end

"""
    rot(q::EdgeData) -> NodeData

Compute the discrete rot of edge data `q`, returning node
data on the same grid. Can also be called as `curl(q)`.
"""
function rot(q::EdgeData{NX,NY}) where {NX,NY}
    w = NodeData(q)
    # Loop over all nodes
    for j in 1:NY+1, i in 1:NX+1
      w.data[i,j] = q.qx[i,j] - q.qx[i,j+1] + q.qy[i+1,j] - q.qy[i,j]
    end
    return w
end

# We can also call this curl, if we feel like it...
curl(q::EdgeData) = rot(q)

# DEVELOP GRAD AND CURL and LAPLACIANS FOR ALL TYPES
