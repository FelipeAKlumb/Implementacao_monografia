using LinearAlgebra
using JuMP
using HiGHS


function encontrar_ponto_viavel(A, b)

    # Encontra um ponto no conjunto víavel dado pelas restrições lineares de desigualdade A * x <= b

    p, n = size(A)  # p: número de restrições, n: número de variáveis originais

    modelo = Model(HiGHS.Optimizer)  # Usa o solver HiGHS
    
    # Variáveis originais x
    @variable(modelo, x[1:n])
    
    # Variáveis de folga s >= 0
    @variable(modelo, s[1:p] >= 0)
    
    # Variáveis artificiais w >= 0
    @variable(modelo, w[1:p] >= 0)
    
    # Função objetivo da Fase 1: minimizar soma das artificiais
    @objective(modelo, Min, sum(w))
    
    # Restrições: transformamos A*x <= b em A*x + s = b

    for i in 1:p

        @constraint(modelo, sum(A[i, j] * x[j] for j in 1:n) + s[i] - w[i] == b[i])

    end

    # Resolver a Fase 1
    optimize!(modelo)

    # Verificar se foi possível encontrar um ponto viável

    if objective_value(modelo) > 1e-6  # Se w[i] > 0, o problema original é inviável

        println("Não existe ponto viável.")
        return nothing

    else

        x0 = value.(x)  # Pegamos os valores das variáveis x
        println("Ponto viável encontrado: ", x0)
        return x0

    end

end


function is_viavel(x0, A)

    # Verifica se o ponto fornecido é viável para o problema

    Ax = A * x0

    if all(Ax .<= -1 + 1e-6)

        return true

    else

        return false

    end

end


function f(w)

    return 0.5 * (w[1]^2 + w[2]^2)

end


function grad_f(w)

    return [w[1], w[2], 0]

end


function grad_res(linha, matriz)
    
    return matriz[linha, :]

end


#=
function proj_restrição(x, linha, matriz)
    
    # Projeta o ponto x no hiperplano da restrição na i-ésima linha da matriz
    
    Ai = matriz[linha, :]

    alfa = - (Ai' * x + 1) / norm(Ai)^2

    return x + alfa * grad_res(linha, matriz)

end
=#


function tentar_resolver_sistema(A, b)

    # Defeito do programa: se nosso conjunto de dados possui dois pontos com coordenadas iguais, haverá linhas de AI Linearmente Dependentes.
    # A função tentar_resolver_sistema retornará :sem_solução_única.
    # Caso seja usado um critério de parada do tipo ||xk+1 - xk|| < epsilon, esse problema pode ser contornado, pois,
    # usando o método do gradiente projetado no núcleo, temos que o d a ser calculado será o vetor nulo.

    try

        x = A \ b  # Tenta resolver o sistema. Um erro ocorrerá caso o sistema seja subdeterminado.

        residuo = norm(A * x - b)  # Como julia automaticamente minimiza a norma de A * x - b se o sistema é sobredeterminado, calculamos o resíduo.

        if residuo > 1e-5  # Se o resíduo é não nulo, o sistema não possui solução.

            return :sem_solucao

        else  # Se o resísuo é nulo, então x é a solução do sistema.

            return x

        end

    catch e  # Se ocorre um erro, o sistema é subdeterminado e possui infinitas soluções

        return :infinitas_solucoes

    end

end


function calcular_alfa_barra(x, d)

    # Calcula um alfa_barra que garanta a nossa permanência no conjunto viável

    possíveis_alfas = []

    for i in P

        if grad_res(i, A)' * d > 1e-6    # Para evitar erros numéricos

            alfa = - (A[i, :]' * x + 1) / (A[i, :]' * d)
            push!(possíveis_alfas, alfa)

        end

    end

    alfa_barra = minimum(possíveis_alfas)
    return alfa_barra

end


function Armijo(x, d, t_barra, alfa=1e-4, max_iter=32)

    # Calcula o passo tk que garante descréscimo suficiente (condição de Armijo) para uma direção de descida d a partir do ponto x

    tk = t_barra
    salvaguarda = 0

    while f(x + tk * d) > f(x) + alfa * tk * grad_f(x)' * d && salvaguarda < max_iter

        tk *= 1 / 2

        salvaguarda += 1

    end

    if salvaguarda == max_iter

        return "Descréscimo suficiente não foi alcançado!"

    else

        return tk

    end

end


function índices_restrições_ativas(x, matriz)

    # Retorna um conjunto com todos os índices das restrições que são ativas em x

    I = []
    Ax = matriz * x

    for (i, v) in enumerate(Ax)

        if -1.001 < v < -0.999    # valor = -1

            push!(I, i)

        end

    end

    return I
    
end


function índ_mult_Lag_não_negativos(mu)

    # Encontra os índices dos multiplicadores de Lagrange não negativos do vetor mu

    índices_multiplicadores_não_negativos = []

    for (i, valor) in enumerate(mu)

        if valor > -1e-3    # Para evitar erros numéricos

            push!(índices_multiplicadores_não_negativos, i)

        end

    end

    return índices_multiplicadores_não_negativos

end


function projeção_grad_f_núcleo(matriz, x, beta=0.1)

    # Calcula uma direção viável e de descida no núcleo da matriz por meio da projeção de - grad_f

    aux_1 = - matriz * grad_f(x)
    aux_2 = (matriz * matriz') \ aux_1
    aux_3 = matriz' * aux_2

    d = - grad_f(x) - aux_3  # Método do gradiente projetado

    norma_d = norm(d)
    norma_grad = norm(grad_f(x))

    if norma_d < beta * norma_grad  # Condição beta para a escolha da direção (importante quando há menos de n + 1 vetores suporte)

        d = beta * (norma_grad / norma_d) * d

    end

    return d

end





# Matriz das restrições de desigualdade

# Conjunto de dados Íris (com 6 observações). Ponto ótimo: [-2.271983, 0.908764, 9.406214]

A = [
    [-5.1 -3.5 -1.0]; # Restrição ativa no minimizador
    [-4.9 -3.0 -1.0]; # Restrição ativa no minimizador
    [-4.7 -3.2 -1.0];
    [7.0 3.2 1.0];
    [6.4 3.2 1.0];
    [5.7 2.8 1.0]; # Restrição ativa no minimizador
]


# Conjunto de dados (fictício) cuja solução possui 2 vetores suporte. Ponto ótimo: [-0.25, 0.25, 1.75]

A = [
    [-15.0 -25.0 -1.0];
    [-10.0 -20.0 -1.0];
    [-12.0 -23.0 -1.0];
    [-15.0 -21.0 -1.0];
    [-19.0 -16.0 -1.0]; # Restrição ativa no minimizador
    [23.0 12.0 1.0]; # Restrição ativa no minimizador
    [25.0 5.0 1.0];
    [30.0 10.0 1.0];
    [30.0 5.0 1.0];
    [35.0 5.0 1.0]
]


# Conjunto de dados Íris (com 100 observações). Ponto ótimo: [-6.315546, 5.262622, 17.316051]

A = [
    [-5.1 -3.5 -1.0];  # Início da classe setosa
    [-4.9 -3.0 -1.0];
    [-4.7 -3.2 -1.0];
    [-4.6 -3.1 -1.0];
    [-5.0 -3.6 -1.0];
    [-5.4 -3.9 -1.0];
    [-4.6 -3.4 -1.0];
    [-5.0 -3.4 -1.0];
    [-4.4 -2.9 -1.0];
    [-4.9 -3.1 -1.0];
    [-5.4 -3.7 -1.0];
    [-4.8 -3.4 -1.0];
    [-4.8 -3.0 -1.0];
    [-4.3 -3.0 -1.0];
    [-5.8 -4.0 -1.0];
    [-5.7 -4.4 -1.0];
    [-5.4 -3.9 -1.0];
    [-5.1 -3.5 -1.0];
    [-5.7 -3.8 -1.0];
    [-5.1 -3.8 -1.0];
    [-5.4 -3.4 -1.0];
    [-5.1 -3.7 -1.0];
    [-4.6 -3.6 -1.0];
    [-5.1 -3.3 -1.0];
    [-4.8 -3.4 -1.0];
    [-5.0 -3.0 -1.0];
    [-5.0 -3.4 -1.0];
    [-5.2 -3.5 -1.0];
    [-5.2 -3.4 -1.0];
    [-4.7 -3.2 -1.0];
    [-4.8 -3.1 -1.0];
    [-5.4 -3.4 -1.0];
    [-5.2 -4.1 -1.0];
    [-5.5 -4.2 -1.0];
    [-4.9 -3.1 -1.0];
    [-5.0 -3.2 -1.0];
    [-5.5 -3.5 -1.0]; # Restrição ativa no minimizador
    [-4.9 -3.6 -1.0];
    [-4.4 -3.0 -1.0];
    [-5.1 -3.4 -1.0];
    [-5.0 -3.5 -1.0];
    [-4.5 -2.3 -1.0]; # Restrição ativa no minimizador
    [-4.4 -3.2 -1.0];
    [-5.0 -3.5 -1.0];
    [-5.1 -3.8 -1.0];
    [-4.8 -3.0 -1.0];
    [-5.1 -3.8 -1.0];
    [-4.6 -3.2 -1.0];
    [-5.3 -3.7 -1.0];
    [-5.0 -3.3 -1.0];
    [7.0 3.2 1.0];  # Início da classe versicolor
    [6.4 3.2 1.0];
    [6.9 3.1 1.0];
    [5.5 2.3 1.0];
    [6.5 2.8 1.0];
    [5.7 2.8 1.0];
    [6.3 3.3 1.0];
    [4.9 2.4 1.0]; # Restrição ativa no minimizador
    [6.6 2.9 1.0];
    [5.2 2.7 1.0];
    [5.0 2.0 1.0];
    [5.9 3.0 1.0];
    [6.0 2.2 1.0];
    [6.1 2.9 1.0];
    [5.6 2.9 1.0];
    [6.7 3.1 1.0];
    [5.6 3.0 1.0];
    [5.8 2.7 1.0];
    [6.2 2.2 1.0];
    [5.6 2.5 1.0];
    [5.9 3.2 1.0];
    [6.1 2.8 1.0];
    [6.3 2.5 1.0];
    [6.1 2.8 1.0];
    [6.4 2.9 1.0];
    [6.6 3.0 1.0];
    [6.8 2.8 1.0];
    [6.7 3.0 1.0];
    [6.0 2.9 1.0];
    [5.7 2.6 1.0];
    [5.5 2.4 1.0];
    [5.5 2.4 1.0];
    [5.8 2.7 1.0];
    [6.0 2.7 1.0];
    [5.4 3.0 1.0]; # Restrição ativa no minimizador
    [6.0 3.4 1.0];
    [6.7 3.1 1.0];
    [6.3 2.3 1.0];
    [5.6 3.0 1.0];
    [5.5 2.5 1.0];
    [5.5 2.6 1.0];
    [6.1 3.0 1.0];
    [5.8 2.6 1.0];
    [5.0 2.3 1.0];
    [5.6 2.7 1.0];
    [5.7 3.0 1.0];
    [5.7 2.9 1.0];
    [6.2 2.9 1.0];
    [5.1 2.5 1.0];
    [5.7 2.8 1.0]
]






# Número de restrições total do problema
p = size(A, 1)

# Vetor do R^{p} composto apenas por -1
b = - ones(p)

# "Conjunto" com índices de todas as restrições
P = 1:p





# Inicia o algoritmo com um ponto viável (calculado pelo método simplex)
x0 = encontrar_ponto_viavel(A, b)

# Sugestões de outros valores para x0:

# Conjunto de dados Iris (setosa e versicolor) com 6 observações:
# x0 = [-5.0, 9.0, -1.0]
# x0 = [-3.7501, 0.036, 20.0]

# Conjunto de dados fictícios com apenas dois vetores suporte:
# x0 = [-20.0, 35.0, 10.0]
# x0 = [-9.0, 15.0, 20.0]

# Conjunto de dados Iris (setosa e versicolor) com 100 observações:
# x0 = [-10.56, 9.5, 27.2]
# x0 = [-18.0, 15.0, 51.0]

# Verifica se, de fato, o ponto x0 é viável
is_viavel(x0, A)

# Mostra quais restrições estão ativas no ponto inicial
println(índices_restrições_ativas(x0, A))


# Inicialização do algoritmo
x = x0

# Critérios de parada
epsilon = 1e-6
fx_novo = 100
fx_antigo = 0

max_iter = 100
salvaguarda = 0


# Método das Restrições Ativas

while salvaguarda < max_iter && norm(fx_novo - fx_antigo) > epsilon

    # Cria o conjunto das restrições ativas no ponto
    I = índices_restrições_ativas(x, A)

    if I == []    # I é vazio

        # A verificação acerca do gradiente ser nulo não é necessária. O gradiente só se anula em vetores do tipo [0, 0, b], com b real.
        # No nosso problema, esses pontos nunca são viáveis, pois não possuem uma reta separadora associada a eles.

        # Passo 7
        dk = - grad_f(x)
            
        # Passo 8
        alfa_barra = calcular_alfa_barra(x, dk)

        # Passo 9
        tk = Armijo(x, dk, alfa_barra)

        fx_antigo = f(x)
        x = x + tk * dk

    else    # I é não vazio

        # Matriz de restrições ativas
        AI = A[I, :]

        # Vetor dos multiplicadores de Lagrange
        mu = tentar_resolver_sistema(AI', - grad_f(x))

        if mu == :sem_solucao

            # Passo 4
            dk = projeção_grad_f_núcleo(AI, x)

            # Passo 5
            alfa_barra = calcular_alfa_barra(x, dk)

            # Passo 6
            tk = Armijo(x, dk, alfa_barra)

            fx_antigo = f(x)
            x = x + tk * dk

        elseif all(mu .>= -1e-4)

            break

        else

            # Encontrar os multiplicadores não negativos para calcular direção viável e de descida
            índices = índ_mult_Lag_não_negativos(mu)

            # Monta a matriz AI_barra, obtida pela retirada das linhas de AI associadas a mult de Lagr negativos
            AI_barra = AI[índices, :]


            # Passo 7
            dk = projeção_grad_f_núcleo(AI_barra, x)
            
            # Passo 8
            alfa_barra = calcular_alfa_barra(x, dk)

            # Passo 9
            tk = Armijo(x, dk, alfa_barra)

            fx_antigo = f(x)
            x = x + tk * dk

        end

    end

    fx_novo = f(x)

    salvaguarda += 1

end

# Ponto obtido:
println(x)

# Restrições ativas no ponto:
println(índices_restrições_ativas(x, A))

# Número de iterações necessárias:
println(salvaguarda)
