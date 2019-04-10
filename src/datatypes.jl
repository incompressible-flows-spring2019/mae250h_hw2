#=
A key for remembering the dimensions of each data type for a grid with
NX, NY interior cells:

CellData: (NX, NY) interior, (NX+2,NY+2) total
          interior indexing 2:NX+1,2:NY+1 (and no boundary)
EdgeData: x component: (NX+1,NY) interior/boundary, (NX+1,NY+2) total
                       so interior indexing 2:NX,2:NY+1
                       and boundary at [1,2:NY+1], [NX+1,2:NY+1]
          y component: (NX,NY+1) interior/boundary, (NX+2,NY+1) total
                       so interior indexing 2:NX+1,2:NY
                       and boundary at [2:NX+1,1], [2:NX+1,NY+1]
NodeData: (NX+1,NY+1) interior/boundary/corner and total (no ghosts)
          interior indexing 2:NX,2:NY
          and boundaries at [1,2:NY], [NX+1,2:NY], [2:NX,1], [2:NX,NY+1]
          and corners at [1,1], [NX+1,1], [1,NY+1], [NX+1,NY+1]

Also, the index spaces of each of these variables are a little different from
each other. Treating Ic as a continuous variable in the cell center index space, and
In as a continuous variable in node index space, then our convention here is that

        i_c = i_n + 1/2

In other words, a node with index i_n = 1 corresponds to a location in the cell
center space at i_c = 3/2.
=#

# We import the + and - operations from Julia so that we can extend them to
# our new data types here
import Base:+, -,*,/

#================= THE DATA TYPES =================#

# This is the parent of all of the grid data types
abstract type GridData{NX,NY} end

#= Here we are constructing a data type for data that live at the cell
centers on a grid.

The NX and NY written inside {} are called the parameters for the type.
These parameters will hold the number of cells of the underlying grid (interior)
in each direction. We can then use these parameters to ensure that all operations
are carried out only on data that correspond to the same size grid.

We are declaring this as a subtype of `GridData{NX,NY}`. GridData{NX,NY} will
be the `parent` type of all data on the grid of size NX x NY.
=#
struct CellData{NX,NY} <: GridData{NX,NY}
  data :: Array{Float64,2}
end

#=
Here we are constructing a data type for cell edges. Notice that we use
NX and NY again as parameters. These still correspond to the number of interior
cells, so that edge data associated with other data types on the same grid
share the same parameter values. We have separate arrays for the x and y
components. Again, it is a subtype of GridData{NX,NY}.
=#
struct EdgeData{NX,NY} <: GridData{NX,NY}
  qx :: Array{Float64,2}
  qy :: Array{Float64,2}
end

#=
Node-based data. Again, NX and NY parameters correspond to number of
interior cells. Again, it is a subtype of GridData{NX,NY}.
=#
struct NodeData{NX,NY} <: GridData{NX,NY}
  data :: Array{Float64,2}
end



#================  CELL-CENTERED DATA CONSTRUCTORS ====================#

# This is called a constructor function for the data type. Here, it
# simply fills in the parameters NX and NY and returns the data in the new type.
"""
    CellData(data)

Set up a type of data that sit at cell centers. The `data` include the
interior cells and the ghost cells, so the resulting grid will be smaller
by 2 in each direction.

Example:
```
julia> w = ones(5,4);

julia> HW2.CellData(w)
HW2.CellData{3,2}([1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0; … ; 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0])
```
"""
function CellData(data::Array)
  #NX,NY = size(data)
  #nx_pad, ny_pad = NX+2, NY+2
  #padded_data = zeros(nx_pad,ny_pad)
  #padded_data[2:nx_pad-1,2:ny_pad-1] = data
  nx_pad, ny_pad = size(data)
  NX, NY = nx_pad-2, ny_pad-2

  return CellData{NX,NY}(data)
end

#== Some other constructors for cell data ==#

# This constructor function allows us to just specify the interior grid size.
# It initializes with zeros
"""
    CellData(nx,ny)

Set up cell centered data equal to zero on a grid with `(nx,ny)` interior cells.
Pads with a layer of ghost cells on all sides.
"""
CellData(nx::Int,ny::Int) = CellData{nx,ny}(zeros(nx+2,ny+2))

# This constructor function is useful when we already have some grid data
# It initializes with zeros
"""
    CellData(p::GridData)

Set up cell centered data equal to zero on a grid corresponding to supplied
grid data `p`. Pads with a layer of ghost cells on all sides.
"""
CellData(p::GridData{NX,NY}) where {NX,NY} = CellData{NX,NY}(zeros(NX+2,NY+2))

# Here we extend the function `size` that is part of Julia to work on our new data type.
# The `where {NX,NY}` addition at the end just allows NX and NY to be any value.
# In other words, this function will work on CellData on any size of grid.
# Note that this returns the total size, including ghost cells.
Base.size(p::CellData{NX,NY}) where {NX,NY} = size(p.data)

#================  EDGE DATA CONSTRUCTORS ====================#

"""
    EdgeData(nx,ny)

Set up edge data equal to zero on a grid with `(nx,ny)` interior cells. Pads
with ghosts where appropriate.
"""
EdgeData(nx::Int,ny::Int) = EdgeData{nx,ny}(zeros(nx+1,ny+2),zeros(nx+2,ny+1))

"""
    EdgeData(p::GridData)

Set up edge data equal to zero on a grid of a size corresponding to the
given grid data `p`. Pads with ghosts.

Example:
```
julia> p = HW2.CellData(5,4)
HW2.CellData{5,4}([0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0])

julia> q = HW2.EdgeData(p)
HW2.EdgeData{5,4}([0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0])
```
"""
function EdgeData(p::GridData{NX,NY}) where {NX,NY}
  nx_qx, ny_qx = NX+1, NY+2
  nx_qy, ny_qy = NX+2, NY+1

  return EdgeData{NX,NY}(zeros(nx_qx,ny_qx),zeros(nx_qy,ny_qy))
end

# If the arrays themselves for qx and qy are provided:
function EdgeData(qx::Array{Float64,2},qy::Array{Float64,2})
  nx_qx, ny_qx = size(qx)
  nx_qy, ny_qy = size(qy)
  NX, NY = nx_qx-1,ny_qx-2
  # Need to make sure these qx and qy are compatible with each other:
  @assert NX == nx_qy-2 && NY == ny_qy-1
  return EdgeData{NX,NY}(qx,qy)
end

#================  NODE DATA CONSTRUCTORS ====================#

"""
    NodeData(nx,ny)

Set up node data equal to zero on a grid with `(nx,ny)` interior cells. Note
that node data has no ghosts.
"""
NodeData(nx::Int,ny::Int) = NodeData{nx,ny}(zeros(nx+1,ny+1))

"""
    NodeData(p::GridData)

Set up node data equal to zero on a grid of a size corresponding to the
given grid data `p`.

Example:
```
julia> p = HW2.CellData(5,4)
HW2.CellData{5,4}([0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0])

julia> w = HW2.NodeData(p)
HW2.NodeData{5,4}([0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0])
```
"""
NodeData(p::GridData{NX,NY}) where {NX,NY} = NodeData{NX,NY}(zeros(NX+1,NY+1))

# if the data themselves are provided as an array
function NodeData(data::Array{Float64,2})
  nxnode, nynode = size(data)
  return NodeData{nxnode-1,nynode-1}(data)
end

# Extend size function to node data
Base.size(p::NodeData{NX,NY}) where {NX,NY} = size(p.data)

#====== SOME BASIC THINGS WE SHOULD BE ABLE TO DO ON THIS DATA =======#

#=
We need to explicitly tell Julia how to do some things on our new data types
=#

# Set it to negative of itself
function (-)(p::Union{CellData,NodeData})
  p.data .= -p.data
  return p
end

function (-)(p::EdgeData)
  p.qx .= -p.qx
  p.qy .= -p.qy
  return p
end

# Add and subtract the same type
function (-)(p1::T,p2::T) where {T<:Union{CellData,NodeData}}
  return T(p1.data .- p2.data)
end

function (+)(p1::T,p2::T) where {T<:Union{CellData,NodeData}}
  return T(p1.data .+ p2.data)
end

function (-)(p1::T,p2::T) where {T<:EdgeData}
  return T(p1.qx .- p2.qx, p1.qy .- p2.qy)
end

function (+)(p1::T,p2::T) where {T<:EdgeData}
  return T(p1.qx .+ p2.qx, p1.qy .+ p2.qy)
end

# Multiply and divide by a constant
function (*)(p::T,c::Number) where {T<:Union{CellData,NodeData}}
  return T(c*p.data)
end


function (/)(p::T,c::Number) where {T<:Union{CellData,NodeData}}
  return T(p.data ./ c)
end

function (*)(p::T,c::Number) where {T<:EdgeData}
  return T(c*p.qx,c*p.qy)
end

(*)(c::Number,p::T) where {T<:GridData} = *(p,c)

function (/)(p::T,c::Number) where {T<:EdgeData}
  return T(p.qx ./ c, p.qy ./ c)
end
