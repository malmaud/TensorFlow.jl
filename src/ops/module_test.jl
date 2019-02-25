module M
    export x
    x=1
    module Y
        using ..M
        function f(y)
            return y+x
        end
    end
end
