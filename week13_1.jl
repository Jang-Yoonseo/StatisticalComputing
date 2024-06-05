### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 5e629d25-6118-444b-a59d-021da400f759
using PlutoUI,Plots,HTTP,CSV,DataFrames,LinearAlgebra,Statistics,Random,Distributions

# ╔═╡ 83e7bec4-1c6d-11ef-3f2b-591f94a974c8
md"""
# 13wk-1: 변환을 의미하는 행렬
"""

# ╔═╡ 974408b1-8167-41f8-bb1f-eb9b39838a79
md"""
## 1. 강의영상
"""

# ╔═╡ 7bf0dbd6-6ea2-41f2-8677-e57317a6f31e
# html"""
# <div style="display: flex; justify-content: center;">
# <div  notthestyle="position: relative; right: 0; top: 0; z-index: 300;">
# <iframe src=
# "
# https://youtube.com/embed/playlist?list=PLQqh36zP38-zN5HD4xmAMyhxTGr7RY8VY&si=j2GhMZ9uo0TcbGuk
# "
# width=600 height=375  frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
# """

# ╔═╡ ae425091-985a-4fac-b6b5-57e7c5d3a144
md"""
## 2. Imports
"""

# ╔═╡ 190e9d94-30b9-4a9f-9408-6325c6839590
PlutoUI.TableOfContents()

# ╔═╡ fc43f2b3-9c5a-4ad5-be76-6ab36ba170db
Plots.plotly()

# ╔═╡ 3f064ea6-f4d7-4021-8238-b4068c8474b5
md"""
## 3. 변환을 의미하는 행렬
"""

# ╔═╡ b2bcafa5-2d3b-4b96-ae41-42860f32919e
md"""
### A. 열 단위 변환 (column-wise transform)
"""

# ╔═╡ dde0da2a-dc19-415d-b6f1-9139024a99ca
md"""
-- 변환을 의미하는 행렬 ${\bf A}$가 데이터를 의미하는 행렬 ${\bf X}$ 앞에 곱해지는 경우 ${\bf A}$는 ${\bf X}$의 열별로 적용되는 어떠한 선형변환을 의미한다.
"""

# ╔═╡ d9a13225-5c3b-4f61-aff1-a533ceccf994
md"""
${\bf A} {\bf X} = \begin{bmatrix} {\bf A}{\boldsymbol X}_1 & {\bf A}{\boldsymbol X}_2 & \dots & {\bf A}{\boldsymbol X}_p\end{bmatrix}$
"""

# ╔═╡ c805ca53-80e0-4b4f-b008-770153c59642
md"""
-- 예시1: 열별로 평균계산
"""

# ╔═╡ a648c2a0-e2d5-4215-a73a-49c739eca30c
let
	n = 6
	X1 = [0,0,0,10,10,10]
	X2 = [-10,-10,-10,0,0,0]
	X = [X1 X2]
	j = ones(n)
	(1/n*j')*X ≈ [mean(X1) mean(X2)]
end 

# ╔═╡ 91e05f69-5f30-4216-a14e-f563c2d6fe2e
md"""
-- 예시2: 센터링
"""

# ╔═╡ 121e6036-6655-455a-b3d9-0f59b940258b
let
	n = 6
	X1 = [0,0,0,1,1,1]
	X2 = [1,1,1,0,0,0]
	X = [X1 X2]
	j = ones(n)
	J = j*j'
	(I-1/n*J)*X ≈ [X1.-mean(X1) X2.-mean(X2)]
end 

# ╔═╡ 3dad080a-0849-415d-a738-5c4035d0ce3e
md"""
-- 예시3: 공분산행렬
"""

# ╔═╡ df49f281-7275-485e-a943-ca628ec539ac
let
	n = 100
	X1 = randn(n)
	X2 = randn(n)*2 .+ 1 
	X = [X1 X2]
	j = ones(n)
	J = j*j'
	X'*(I-1/n*J)*X/(n-1) ≈ cov(X)
end 

# ╔═╡ 76b7160c-a5ad-4607-8b4e-c472119d7f17
md"""
-- 예시4: 상관행렬
"""

# ╔═╡ a8695603-49d2-407e-935e-40bbe2fdc9cd
let
	Random.seed!(43052)
	n = 100
	X1 = randn(n)
	X2 = randn(n)*2 .+ 1 
	X = [X1 X2]
	j = ones(n)
	J = j*j'
	D = Diagonal(.√diag(cov(X)))
	inv(D)*cov(X)*inv(D) ≈ cor(X)
end 

# ╔═╡ a536bd0c-0b64-42b6-914d-ea46deadb5de
md"""
-- 예시5: SST
"""

# ╔═╡ e712ebff-1f70-48e0-a65d-a27378296a16
let
	n = 100
	y = randn(n)
	j = ones(n)
	J = j*j'
	y'*(I-J/n)*y ≈ sum((y .- mean(y)).^2)
end 

# ╔═╡ a7f3d892-1be5-4b96-8aca-dd25a858da4c
md"""
-- 예시6: 이동평균
"""

# ╔═╡ 59eb5cc9-4a13-4b56-8044-d242d3f583ae
k = @bind k Slider(0:20,show_value=true)

# ╔═╡ 2b025cbb-7d49-4a03-a0c4-73e96ff6502a
let 
	Random.seed!(43052)
	n = 100
	t = (1:n)/n
	y = sin.(2π*t) + randn(n)*0.2
	scatter(t,y,alpha=0.2,color="gray")
	dl,d,du = ones(n-1)/3, ones(n)/3, ones(n-1)/3
	M = Tridiagonal(dl,d,du) #구현하고자 하는 행렬을 만들어주는 기능.. 따로 공부해야할듯.
	plot!(t,(M^k)*y,linewidth=2, color="red") #M이 선형변환을 해주니까 가능한 결과.
end 

# ╔═╡ 9c8779e6-3ee1-45c0-a2bc-cd2e937ee217
md"""
### B. 행 단위 변환 (row-wise transform)
"""

# ╔═╡ 0b3104d1-df93-414b-ba29-f5ee2f162234
md"""
-- 변환을 의미하는 행렬 ${\bf A}$가 데이터를 의미하는 행렬 ${\bf X}$ 뒤에 곱해지는 경우 ${\bf A}$는 ${\bf X}$의 행별로 적용되는 어떠한 선형변환을 의미한다.
"""

# ╔═╡ 8fa00f0b-ba3b-4a49-821c-87ecc607c1d3
md"""
${\bf X}{\bf A} = \begin{bmatrix} {\boldsymbol x}^\top_1{\bf A} \\ {\boldsymbol x}^\top_2{\bf A} \\ \dots \\ {\boldsymbol x}^\top_n{\bf A}\end{bmatrix}$
"""

# ╔═╡ acc4ba11-477f-4f7c-a7d1-c0f115425a4e
begin
	url = "https://raw.githubusercontent.com/guebin/2021IR/master/_notebooks/round2.csv"
	df = CSV.File(download(url))
	X1,X2 = df.x[1:30:end], df.y[1:30:end]
	X = [X1 X2]
	scatter(
		X1,X2,
		xlim=(-1000,2000),ylim=(-750,1500),
	)
end

# ╔═╡ 9828643f-9e51-45da-a1c9-7b9cdbfc9daa
md"""
-- 예시1: 스케일링
"""

# ╔═╡ 4c2acf0a-ecbd-4d4f-9fb7-8c1bb2a44893
let
	A = [2.0 0 
		 0   0.5] #x축은 두배 길어지고, y축은 1/2배..
	Z = X*A
	Z1,Z2 = eachcol(Z)
	scatter(X1,X2, xlim=(-1000,2000),ylim=(-750,1500))	
	scatter!(Z1,Z2)
end 

# ╔═╡ af6c1445-b7ed-4fa9-8465-32af890da919
md"""
-- 예시2: 사영
"""

# ╔═╡ 1d257e08-68fb-449e-aed7-64b23047ec23
let
	A = [1 0 
		 0 0] #값 유지. y는 0 으로 눌러버림.
	Z = X*A
	Z1,Z2 = eachcol(Z)
	scatter(X1,X2, xlim=(-1000,2000),ylim=(-750,1500))	
	scatter!(Z1,Z2)
end 

# ╔═╡ 9276a1a1-c7d4-47c6-841a-7ba7b0678b47
md"""
-- 예시3: 대칭이동
"""

# ╔═╡ 67dcaab6-c244-48ad-b354-c073faf41fa8
let
	Random.seed!(43052)
	A = [-1  0 
		  0 -1]
	Z = X*A
	Z1,Z2 = eachcol(Z)
	scatter(X1,X2, xlim=(-1000,2000),ylim=(-750,1500))	
	scatter!(Z1,Z2)
end 

# ╔═╡ e079a51a-92a3-44f5-af30-994f4c7c6c3e
md"""
-- 예시4: 회전이동
"""

# ╔═╡ 9d8c4cbe-5b21-4633-b5ef-3bd7a78e0c4b
theta = @bind θ Slider(-π:(2π/24):π,show_value=true,default= π/3) #Sider 가운데는 간격.

# ╔═╡ 8797fe9b-879a-4d54-ac12-55a56b93b0a5
let 
	A = [ cos(θ) sin(θ)
		 -sin(θ) cos(θ)]
	Z = X*A #각 옵저베이션에 대해 외전변환... 따로따로 보는 관점을 생각하는 것..
	Z1,Z2 = eachcol(Z)
	scatter(X1,X2, xlim=(-1000,2000),ylim=(-750,1500))	
	scatter!(Z1,Z2)
end 

# ╔═╡ 92426a94-bf38-4e54-b8a1-a69c1bb1cb0a
md"""
-- 예시5: 디커플링
"""

# ╔═╡ 4ad6d6fa-7345-4459-ae2d-b3321fc7a76d
let 
	Random.seed!(43052)
	n = 500
	μ = [0,0]
	Σ = [1.0 0.7
		 0.7 2.0]
	X = rand(MvNormal(μ,Σ),n)'
	X1,X2 = eachcol(X)
	p1 = scatter(X1,X2,alpha=0.2,xlim=(-7,7),ylim=(-7,7),label="X")
	Σ̂ = cov(X) #모 공분산 행렬의 추정량.
	λ,Ψ = eigen(Σ̂) #대칭행렬은 고유분해 가능.
	A = Ψ*Diagonal(.√(1 ./λ))
	Z = X*A 
	Z1,Z2 = eachcol(Z)
	p2 = scatter(Z1,Z2,alpha=0.2,xlim=(-7,7),ylim=(-7,7),label="Z")
	plot(p1,p2) #이론적 설명 필기따기 ㅋ.
end 

# ╔═╡ 1632539f-f0be-4640-bbc3-a113f0169d84
md"""
## 4. 직교행렬과 변환
"""

# ╔═╡ d59c37b5-a201-4220-8a76-aadc50c343eb
md"""
### A. 직교변환의 특징
"""

# ╔═╡ fd9ae9d3-20a4-4765-9935-484d04ec09f3
md"""
-- 어떠한 벡터에 직교행렬이 변환으로 적용되면, 그 벡터는 크기와 각도가 보존된다.

**열벡터에 적용되는 경우**
- 크기보존: $\|{\boldsymbol X}_1\|^2=\|{\bf A} {\boldsymbol X}_1\|^2$ 이므로 크기가 보존 
- 각도보전: $\frac{{\boldsymbol X}_1 \cdot {\boldsymbol X}_2}{\|{\boldsymbol X}_1\|\|{\boldsymbol X}_2\|}=\frac{({\bf A}{\boldsymbol X}_1)\cdot ({\bf A}{\boldsymbol X}_2)}{\|{\bf A}{\boldsymbol X}_1\|\|{\bf A}{\boldsymbol X}_2\|}$ 이므로 각도도 보존 

**행벡터에 적용되는 경우**
- 크기보존: $\|{\boldsymbol x}_1^\top\|^2=\|{\boldsymbol x}_1^\top {\bf A} \|^2$ 이므로 크기가 보존 
- 각도보전: $\frac{{\boldsymbol x}_1^\top \cdot {\boldsymbol x}_2^\top}{\|{\boldsymbol x}_1^\top\|\|{\boldsymbol x}_2^\top\|}=\frac{({\boldsymbol x}_1^\top{\bf A})\cdot ({\boldsymbol x}_2^\top{\bf A})}{\|{\boldsymbol x}_1^\top {\bf A}\|\|{\boldsymbol x}_2^\top{\bf A}\|}$ 이므로 각도도 보존 

"""

# ╔═╡ b8584068-76b5-4205-8de7-ddbfc6223f07
md"""
### B. 직교행렬(=직교변환)의 예시
"""

# ╔═╡ e47e3400-35d8-4849-b624-4aafc8cae2de
md"""
-- 예시1: 회전
"""

# ╔═╡ 2e94478d-af51-48e6-b908-cce77ef9f35b
let 
	A = [ cos(θ) sin(θ)
		 -sin(θ) cos(θ)]
	@show A'A ≈ I
	Z = X*A
	Z1,Z2 = eachcol(Z)
	i,j = 1,3
	X[i,:]'X[j,:] ≈ Z[j,:]'Z[i,:]
end 

# ╔═╡ f7018fc2-1cc6-4674-a40e-85081e625616
md"""
-- 예시2: 반사
"""

# ╔═╡ 711bfed6-c3ec-4726-94f8-53d1c389e7cd
let
	A = [-1  0 
		  0 -1]
	@show A'A ≈ I
	Z = X*A
	i,j = 1,3
	X[i,:]'X[j,:] ≈ Z[j,:]'Z[i,:]
end 

# ╔═╡ e48c2c42-cb87-48f6-a4e7-8fb38f4961c6
md"""
!!! info "직교변환에 대한 성질"
	아래의 성질을 기억하면 유용하다. 
	- 모든 직교변환 ${\bf A}$는 (1) rotation matrix (2) non-rotational matrix 로 구분할 수 있다. 
	- 2차원에 한정한다면 직교변환은 (즉 ${\bf A}$ 가 $2\times 2$ matrix 일 경우) 항상 (1) rotaion matrix 이거나 (2) reflection matrix 이다. 
	- 1차원에 한정한다면 직교변환은 $\times 1$ 혹은 $\times -1$ 을 의미한다. 
"""

# ╔═╡ 34088cda-a940-4042-8bbd-278e8b0407d2
md"""
!!! warning "직교변환의 느낌"
	직교변환이 데이터를 의미하는 행렬 ${\bf X}$ 뒤에 곱해질경우 각 관측치의 크기 및 각도(=상대적위치=형태)가 모두 보존된다. 따라서 어떠한 자료 ${\bf X}$가 있고 그러한 ${\bf X}$에 적당한 직교변환을 하여 ${\bf Z}$를 얻었다면, 즉 아래가 성립한다면 

	$${\bf Z}={\bf X}{\bf A},\quad {\bf A}^\top{\bf A}={\bf I}$$

	사실상 ${\bf Z}$는 ${\bf X}$와 거의 동등한 자료라고 보면 된다. (조금 틀어서 본것 밖에 없음)
"""

# ╔═╡ 13624ca3-3958-4481-9586-9f9ad5f50d7d
md"""
### C. PCA와 직교변환
"""

# ╔═╡ e543b1f4-693a-478b-a111-9da9279d1032
md"""
-- 그렇다면 아래는 어떰?
"""

# ╔═╡ 9b2b7f7f-cee4-42f6-bf6d-6c9347fdba52
let
	df2 = DataFrame(CSV.File(HTTP.get("https://raw.githubusercontent.com/guebin/SC2024/main/toeic.csv").body))
	X1,X2,X3 = eachcol(df2)
	X = [X1 X2 X3] 
	U,d,V = svd(X)
	Z = X*V
	Z1,Z2,Z3 = eachcol(Z)
	i,j = 1,3
	X[i,:]'X[j,:] ≈ Z[j,:]'Z[i,:]
end 

# ╔═╡ 078a4099-d919-4121-859b-023c2a30edf0
md"""
!!! warning "PCA의 이해"
	임의의 행렬 ${\bf X}$ 에 대한 ${\bf Z}={\bf X}{\bf V}$ 는 ${\bf X}$를 재표현한것 뿐이다. ${\bf X}$의 각 관측치가 가지고 있는 크기, 및 모양이 모두 보존되므로 사실상 ${\bf Z}$는 ${\bf X}$와 같은 데이터라고 볼 수 있다.
"""

# ╔═╡ 456aa9bf-08be-4bfc-82b5-1f51d13f073d
md"""
-- PCA를 해서 얻는 이득: 차원축소의 관점 말고 PCA를 사용해서 얻는 이득이 뭘까?
"""

# ╔═╡ Cell order:
# ╟─83e7bec4-1c6d-11ef-3f2b-591f94a974c8
# ╟─974408b1-8167-41f8-bb1f-eb9b39838a79
# ╠═7bf0dbd6-6ea2-41f2-8677-e57317a6f31e
# ╟─ae425091-985a-4fac-b6b5-57e7c5d3a144
# ╠═5e629d25-6118-444b-a59d-021da400f759
# ╠═190e9d94-30b9-4a9f-9408-6325c6839590
# ╠═fc43f2b3-9c5a-4ad5-be76-6ab36ba170db
# ╟─3f064ea6-f4d7-4021-8238-b4068c8474b5
# ╟─b2bcafa5-2d3b-4b96-ae41-42860f32919e
# ╟─dde0da2a-dc19-415d-b6f1-9139024a99ca
# ╟─d9a13225-5c3b-4f61-aff1-a533ceccf994
# ╟─c805ca53-80e0-4b4f-b008-770153c59642
# ╠═a648c2a0-e2d5-4215-a73a-49c739eca30c
# ╟─91e05f69-5f30-4216-a14e-f563c2d6fe2e
# ╠═121e6036-6655-455a-b3d9-0f59b940258b
# ╟─3dad080a-0849-415d-a738-5c4035d0ce3e
# ╠═df49f281-7275-485e-a943-ca628ec539ac
# ╟─76b7160c-a5ad-4607-8b4e-c472119d7f17
# ╠═a8695603-49d2-407e-935e-40bbe2fdc9cd
# ╟─a536bd0c-0b64-42b6-914d-ea46deadb5de
# ╠═e712ebff-1f70-48e0-a65d-a27378296a16
# ╟─a7f3d892-1be5-4b96-8aca-dd25a858da4c
# ╠═59eb5cc9-4a13-4b56-8044-d242d3f583ae
# ╠═2b025cbb-7d49-4a03-a0c4-73e96ff6502a
# ╟─9c8779e6-3ee1-45c0-a2bc-cd2e937ee217
# ╟─0b3104d1-df93-414b-ba29-f5ee2f162234
# ╟─8fa00f0b-ba3b-4a49-821c-87ecc607c1d3
# ╠═acc4ba11-477f-4f7c-a7d1-c0f115425a4e
# ╟─9828643f-9e51-45da-a1c9-7b9cdbfc9daa
# ╠═4c2acf0a-ecbd-4d4f-9fb7-8c1bb2a44893
# ╟─af6c1445-b7ed-4fa9-8465-32af890da919
# ╠═1d257e08-68fb-449e-aed7-64b23047ec23
# ╟─9276a1a1-c7d4-47c6-841a-7ba7b0678b47
# ╠═67dcaab6-c244-48ad-b354-c073faf41fa8
# ╟─e079a51a-92a3-44f5-af30-994f4c7c6c3e
# ╠═9d8c4cbe-5b21-4633-b5ef-3bd7a78e0c4b
# ╠═8797fe9b-879a-4d54-ac12-55a56b93b0a5
# ╟─92426a94-bf38-4e54-b8a1-a69c1bb1cb0a
# ╠═4ad6d6fa-7345-4459-ae2d-b3321fc7a76d
# ╟─1632539f-f0be-4640-bbc3-a113f0169d84
# ╟─d59c37b5-a201-4220-8a76-aadc50c343eb
# ╟─fd9ae9d3-20a4-4765-9935-484d04ec09f3
# ╟─b8584068-76b5-4205-8de7-ddbfc6223f07
# ╟─e47e3400-35d8-4849-b624-4aafc8cae2de
# ╠═2e94478d-af51-48e6-b908-cce77ef9f35b
# ╟─f7018fc2-1cc6-4674-a40e-85081e625616
# ╠═711bfed6-c3ec-4726-94f8-53d1c389e7cd
# ╟─e48c2c42-cb87-48f6-a4e7-8fb38f4961c6
# ╟─34088cda-a940-4042-8bbd-278e8b0407d2
# ╟─13624ca3-3958-4481-9586-9f9ad5f50d7d
# ╟─e543b1f4-693a-478b-a111-9da9279d1032
# ╠═9b2b7f7f-cee4-42f6-bf6d-6c9347fdba52
# ╟─078a4099-d919-4121-859b-023c2a30edf0
# ╟─456aa9bf-08be-4bfc-82b5-1f51d13f073d
