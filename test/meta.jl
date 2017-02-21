using Base.Test

let
    g = Graph()
    local i, j
    as_default(g) do
        @named begin
            i = constant(1)
            j = constant(2)
        end
    end
    @test i == get_tensor_by_name(g, "i")
    @test j == get_tensor_by_name(g, "j")
end
