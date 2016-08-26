# Core operations

## Types

The computation graph as a whole is stored in the `Graph` type.
Individual nodes in the graph are referred to as operations, stored in the `Operation` type.
Operations can produce one or more outputs, represented as the `Tensor` type.

```@docs
Tensor
Operation
Session
Graph
```

## Functions
```@docs
eltype
node_name
get_collection
get_def_graph
run
get_node_by_name
get_shape
get_def
```
