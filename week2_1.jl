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

# ╔═╡ 3e7b4e0d-c52e-4214-a8fb-7ae7b03a60e9
using Distributions, Plots,PlutoUI

# ╔═╡ 05e369d7-e0b8-4ef4-8831-938423a6717e
md"""
# 02wk-1: 베르누이, 이항분포, 포아송
"""

# ╔═╡ db4dd854-1764-4a62-a77d-498ac1c008b1
md"""
## 1. 강의영상
"""

# ╔═╡ 3e2405d2-5c92-42cd-bbc6-66c1add0d282
# html"""
# <div style="display: flex; justify-content: center;">
# <div  notthestyle="position: relative; right: 0; top: 0; z-index: 300;">
# <iframe src=
# "
# https://www.youtube.com/embed/playlist?list=PLQqh36zP38-ycDx8HFZQt_HHG7vLQCHOr
# "
# width=600 height=375  frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
# """

# ╔═╡ 77c4a801-d098-4f45-9b99-4e7e6f55a10b
md"""
## 2. Imports 
"""

# ╔═╡ 5bf2d19f-b8bb-43be-bc35-a963a01f71cb
PlutoUI.TableOfContents() #목차 만들기

# ╔═╡ 6155f666-e228-463e-9721-39ee899366bf
Plots.plotly() #객체지향 plot

# ╔═╡ f94127c6-ca98-4946-babf-cb08768a7b4b
md"""
## 3. 베르누이: $X \sim Bernoulli(p)$ 
"""

# ╔═╡ 74dbc7ae-3293-495b-bb3c-e633bfa76c33
md"""
### A. 기본내용
"""

# ╔═╡ 68ce3dd0-fa08-468d-bd54-1f2f383faab6
md"""
-- 간단한 요약

- X의의미: 성공확률이 p인 1번의 시행에서 성공한 횟수를 X라고 한다. 
- X의범위: X=0 or X=1
- 파라메터의 의미와 범위: p는 성공확률을 나타냄, p ∈ [0,1].
- pdf: 
- mgf: 
- E(X)=p
- V(X)=p(1-p)
"""

# ╔═╡ 5d2716e2-c5d3-470e-8178-500f313172c4
md"""
-- 난수생성코드 (줄리아문법)
"""

# ╔═╡ 239478fc-539d-48de-aba8-73dafcb4a6c9
let 
	p = 0.6 # 파라메터
	N = 10 # 샘플수 
	distribution = Bernoulli(p) # 분포오브젝트 자체를 정의 
	X = rand(distribution,N) # N-samples 분포 자체 객체와 N을 같이 받음..
end

# ╔═╡ 00dbac24-e344-4d79-9b5c-5cf7541d33bb
rand(Bernoulli(0.3) , 10) |> sum #이러면 이항분포 10,3 원소 하나 본것과 같음.

# ╔═╡ 3ca030b1-4de8-4460-a16d-4aae3627c82a
rand(Bernoulli(0.5), 100) |> sum #50 개 쯤 되지 않겠나 기대.

# ╔═╡ 14e43717-d59d-4184-b40b-df6bd0fb94b0
슬라더 = @bind oo Slider(0:0.1:1, show_value = true, default = 0.3)
#슬라이더 객체 생성

# ╔═╡ 8537c3e7-d2f6-41ce-8563-3136317c16d6
oo #숫자가 나옴

# ╔═╡ 5574fb2e-f9a1-47ca-bd19-6ae28a2d69b5
슬라더 # 슬라더 객체가 나옴, 잘 구별할 것!

# ╔═╡ d39cc97b-f7ad-4e54-a6ed-0497d6346acf
md"""
### B. 모수 $\to$ 히스토그램
"""

# ╔═╡ 8dd2007b-dfc3-453a-96c9-7e1df21c871e
md"""
distribution = Bernoulli(p) distribution 이라는 오브젝트는 베르누이 분포라는 객체임.
"""

# ╔═╡ 4beb013b-6540-48b2-9d52-818bf34f7ef1
md" ## 데커레이터 "

# ╔═╡ 710c494e-41cd-4580-ab7d-748516e894bf
### 데커레이터 라는 것을 이해해보자.
Slider(0:0.1:1, show_value = true, default = 0.7) #슬라이더라는 오브젝트가 나옴
#저장은 안됨.

# ╔═╡ eec1ffe4-13da-49d1-8af6-778a543a6bdd
@bind sslider Slider(0:0.1:1, show_value = true, default = 0.7)
#sslider 라는 숫자 객체를 생성해 준 것은 덤..!
#이러한 것을 데커레이터 라고 한다..

# ╔═╡ 15e246db-44ae-48ac-bfa1-a27fca6cb5e2
sslider #숫자가 잘 나오죠?

# ╔═╡ 69cbd1cc-8042-4921-bc49-74420f7f3ae1
md" ## 이어서"

# ╔═╡ a0c5fa5c-4fef-4862-9ce2-895c03eba9d5
md"p11_ = $(@bind p11 Slider(0.1:0.1:0.9, show_value= true, default=0.3))"
#p = @bind p Slider(0.1:0.1:0.9, show_value= true, default=0.3)
#md로 감싸서 코드를 구현할 땐, $( ) 로 감싼다.
#그러면 p 가 문자열로 취급되기 때문에 위에선 p11.. 마크다운으로도 넣을 수 있고
#객체를 구별할 수 있다. 객체의 중복을 피하기 위함..!!
#지금은 아래 예제에다가 p11을 입력할 떄 이름이 아니라 데커레이터 된 놈이 들어간다는 걸
#알기 위함..!!

# ╔═╡ ae9b80c4-eee7-4032-aa5e-ab195fb75722
let #여러 줄 쓰는 법~!
	N = 1000
	histogram(rand(Bernoulli(p11),N))
end
#위의 p11 과 맞물려서 plot이 interactive하게 작동!

# ╔═╡ a302696f-3667-488c-a2c0-a95eca178801
md"""
### C. 난수생성 테크닉
"""

# ╔═╡ 83966cac-2ba7-4751-a571-e0c36840731a
md"""
난수 생성법의 기본 아이디어.. Uniform 분포의 난수는 생성할 수 있다는 것을 안다..\
그럼 나머지 분포들의 난수는 어떻게? 시험에 무조건 낼것이며, 무조건 완벽히 숙지"""

# ╔═╡ b1eb6071-3da2-44ce-bc8e-7ad7f5910bb3
md"""
-- *베르누이분포에서 100개의 샘플을 뽑는 방법 (p=0.37로 가정)*
"""

# ╔═╡ caa9606b-a2b9-4d2b-93c5-453f4525b051
md"""
(방법1) -- 기본
"""

# ╔═╡ a5013797-d2b8-4f53-8f30-0733f248e9b8
md"기본 꼴 : rand(분포,생성시키고 싶은 난수의 개수"

# ╔═╡ b402c405-c6f6-4f53-886b-05b12021b6c3
rand(Bernoulli(0.37),100)

# ╔═╡ bba8738c-2d5c-4d40-bb3d-1a11a57fd9d6
md"""
(방법2) 균등분포 -> 베르누이분포 
"""

# ╔═╡ 74ab3bea-c2d2-4fe3-acd7-7de9b007b7ef
rand(100) # 유니폼에서 100개의 샘플 추출 기본 of 기본.

# ╔═╡ 59de810d-989a-4dca-99d5-2045d8785cfc
rand(100) .< 0.5 # p=0.5인 베르누이 샘플링 개별적인 비교 .<
#이러면 p 가 0.5인 베르누이 분포 난수를 추출한 것과 효과가 같다.

# ╔═╡ 435bbf1e-531b-4a0e-9b69-3b756e2371b3
let
	histogram(rand(100) .< 0.37, color = 2, label = "bernoulli") # 0.37보다 작은것만 성공 그리고 색깔을 지정할 떈 color 라고 해야함... 풀네임 지향!
	#부울 오퍼레이터도 시각화가 가능,, 숫자로 취급!!
	title!("(a)") #느낌표가 붙은 함수는 기존의 것을 조작하겠단 의미
end

# ╔═╡ 27f8e253-01e9-44e6-bb7d-c1de2a8d37c9
md"""
-- 방법1,2의 비교
"""

# ╔═╡ d0d7b51a-9edf-426b-b3e4-cded5f91e6c7
let 
	X = rand(Bernoulli(0.37),1000) # 방법1
	Y = rand(1000) .< 0.37 # 방법2
	p1 = histogram(X,label="X ~ 베르누이",color=1);title!("베르누이 난수")
	p2 = histogram(Y,label="X:=I(Y<p), Y~균등분포",color=2);title!("균등분포 난수")
	plot(p1,p2)
	#플랏 각각에 타이틀을 지정할 수 있다. 
	#이미 있는 플랏에 조작하겠다는 의미로 함수 뒤에 !를 작성
end 

# ╔═╡ 50a1bd66-19eb-41e4-a0e5-4d7369112a4d
md"""
### D. 분산을 최대화
"""

# ╔═╡ 1a2e694b-322d-48e4-b381-ab2cd277366f
md"""
`-` 분산의 그래프 
"""

# ╔═╡ b4b0a042-6936-4dc0-b212-b45b7131d495
plot(p -> p*(1-p),0,1) #이런 식으로도 그림을 그릴 수 있다..
#함수 관계를 작성하고 바로 plot 함수를 씌운 것!!!

# ╔═╡ de89581f-47fd-48e9-a6d9-d22311700e83
let 
	p = 0:0.01:1 
	plot(p, p -> p*(1-p))
end 

# ╔═╡ 78f8e0a8-e1aa-4c70-a460-a179456bd8fc
md"""
## 4. 이항분포: $X \sim B(n,p)$ 
"""

# ╔═╡ 82b80b24-6fb2-43c4-a72c-d0c7b46c403c
md"""
### A. 기본내용

-- 간단한 요약

- X의 의미: 성공확률이 p인 n번의 시행에서 성공한 횟수를 X라고 한다. 
- X의 범위: X=0,1,...,n
- 파라메터의 의미와 범위: n은 시행횟수를 p는 성공할 확률을 의미.  n=1,2,... and p ∈ [0,1].
- pdf:
- mgf: (베르누이분포의mgf)ⁿ -- 왜??
- 평균: np
- 분산: np(1-p)
"""

# ╔═╡ c147f8a3-43b7-447c-9d15-fc6635af4221
md"""
-- 대의적정의 ($\star$)

!!! note "이항분포의 대의적 정의"
	베르누이 분포를 합치면 이항분포가 되며, 이항분포는 베르누이 분포의 합으로 쪼갤 수 있다. 
	 $X \sim B(n,p) \Leftrightarrow X \overset{d}{=} Z_1+Z_2+ \dots +Z_n$, where $Z_i \overset{iid}{\sim} Bernoulli(p)$
"""

# ╔═╡ d74a7832-9481-44fc-9e93-7c4e6dda5664
md"""
-- 난수생성코드 (줄리아문법)
"""

# ╔═╡ 886524f9-447a-4661-adb3-f345314d4750
let 
	p,n = 0.6,30 # 파라메터 언패킹
	N = 10 # 샘플수 
	distribution = Binomial(n,p) # 분포오브젝트 자체를 정의 
	X = rand(distribution,N) # N-samples 
end

# ╔═╡ 3009eb43-63ce-46ec-ad4e-c20acee8e261
N # 실행이 안되는 걸 보니 let end로 묶이는 곳에서만 성립하는 local 변수?
#맞다

# ╔═╡ 9e1dd3ca-7a75-4db8-bed9-8885d0678ad6
md"""
### B. 모수 $\to$ 히스토그램
"""

# ╔═╡ 8dde41a2-7164-4568-9cbe-c91f4e29bf5f
md"p = $(@bind p Slider(0.1:0.1:0.9, show_value=true, default=0.3))"
#p = @bind p Slider(0.1:0.1:0.9, show_value=true, default=0.3)

# ╔═╡ 8ea5c74c-95a8-4980-a144-3f2cf1fb0453
md"n = $(@bind n Slider(1:1:300, show_value = true, default=10))"
#n = @bind n Slider(1:1:30, show_value = true)

# ╔═╡ f5627b38-1bee-4c03-b83c-79b2632d19bd
let
	N = 100
	histogram(rand(Binomial(n,p),N),)
	#xlims!(0,30) #x의 리미트를 수정!
end

# ╔═╡ 33d3125e-3d5f-467c-8488-2e82f9832e51
md"""
### C. 난수생성 테크닉
"""

# ╔═╡ 631466f1-e3a8-4df1-ad1e-255a3b1e6e32
md"""
-- *이항분포에서 100개의 샘플을 뽑는 방법 (p=0.37, n=8 이라고 가정)*
"""

# ╔═╡ d8bb13c9-9ad9-473d-87ed-dacbefc23cf2
rand(Bernoulli(0.37),8) |> sum #점 찍고 안찍고 차이 느끼기.

# ╔═╡ e2abb175-945b-48a7-a806-a05606e92290
[rand(Bernoulli(0.37),8) |> sum for i in 1:100] #리스트 컴프리헨션.
#이항 분포의 대의적 정의를 떠올린다면 하나도 어렵지 않다.

# ╔═╡ 69dd4150-f82a-40a7-a1e1-ab2c90f024dd
md"""
(방법1) 
"""

# ╔═╡ 682bf567-0e62-475c-9078-6d873b6ac911
rand(Binomial(8,0.37),100)

# ╔═╡ 4b7789e6-ff5f-4000-b8ff-7d888d4e2e97
md"""
(방법2) 베르누이 -> 이항분포
"""

# ╔═╡ a2ca922b-432a-4215-b51b-32db273852d1
rand(Bernoulli(0.37),8) #베르누이 시행 8번 -> 더하면 B(8,0.37) 

# ╔═╡ 1c838c8b-cb51-4d7b-87ab-95cb3780ad2b
rand(Bernoulli(0.37),8) |> sum #그냥 한번 해봄

# ╔═╡ aa72e48f-9b28-46d8-9cbc-7cd0c1191879
[rand(Bernoulli(0.37),8) for i in 1:100]

# ╔═╡ e86e0e87-313a-4851-8cc3-00011d7cc252
[rand(Bernoulli(0.37),8) for i in 1:100] .|> sum #이항분포 난수 100개 추출..
#리스트 안에 리스트가 있는데, 그 리스트 하나 하나를 원소로 보고 sum을 진행한 것.

# ╔═╡ 56c84dde-be00-48c4-a074-6f90c39e7092
md"""
(방법3) 균등분포 -> 베르누이분포 -> 이항분포 
"""

# ╔═╡ 3d63d57c-a54f-4571-a514-c686d6415b70
rand(8) # 유니폼에서 8개를 뽑는다. 

# ╔═╡ ce2c1125-8114-4b74-a86a-2dacffd4a811
rand(8) .< 0.37 # 성공확률이 0.37인 베르누이에서 8개의 샘플을 뽑은셈 

# ╔═╡ 12d5fd97-7a1c-405f-87d6-fa5317636852
[rand(8) .< 0.37 for i in 1:100] #성공확률이 0.37 이면... 0.37보다 작은 숫자들에 관심을 기울이자..

# ╔═╡ 8cdee83a-d7bf-4c75-a7c5-35168b474ffc
[sum(rand(8) .< 0.47) for i in 1:100]

# ╔═╡ 87fcf7c3-0659-486f-9924-2978e2fe45b2
[rand(8) .< 0.37 for i in 1:100] .|> sum # (n,p)=(8,0.37)인 이항분포에서 100개를 뽑은셈 

# ╔═╡ 7a474be8-a3f0-4b7f-94ab-90eda88bfefc
md"""
### D. 분산의 최대화
"""

# ╔═╡ d24f2b08-8997-4cc5-8f87-4a5cbd8290f2
md"""
`-` 분산의 그래프 
"""

# ╔═╡ 7e1e6e3f-f13a-414d-9b12-b1a610f61444
let 
	n = 10 
	p = 0:0.01:1 
	plot(p, p-> n*p*(1-p))
end 

# ╔═╡ 6c429ac2-2cca-45a5-a3dc-e09f94275732
md"""
## 5. 포아송분포: $X \sim Poi(\lambda)$
"""

# ╔═╡ 883ceb79-7e3d-496a-b76c-78d0e586c1d9
md"""
### A. 기본내용
"""

# ╔═╡ fb89feac-7ee3-424c-8367-8f4ef577b12c
md"""
-- 간단한 요약
- X의의미: 발생횟수의 평균이 λ인 분포에서 실제 발생횟수를 X라고 한다. 
- X의범위: 발생안할수도 있으므로 X=0이 가능. 따라서 X=0,1,2,3,... 
- 파라메터의 의미와 범위: λ = 평균적인 발생횟수; λ>0. 
- pdf: 
- mgf: 
- E(X): λ
- V(X): λ
"""

# ╔═╡ fc2b7c5b-9cb7-4536-8f55-7a502319b3c6
md"""
-- 난수생성코드(줄리아문법)
"""

# ╔═╡ 8f4f4c31-b997-4bf7-b8ee-77f5c1c90b75
let 
	λ = 5.3 # 파라메터 평균 발생 횟수라서 소수가 가능, 음수만 아니면 됨.
	N = 10 # 샘플수 
	distribution = Poisson(λ) # 분포오브젝트 자체를 정의 
	X = rand(distribution,N) # N-samples 
end

# ╔═╡ 3b13a7d5-d476-4d97-bfef-374bbf136b53
md"""
-- [포아송분포의 예시](https://www.statology.org/poisson-distribution-real-life-examples/#:~:text=Example%201%3A%20Calls%20per%20Hour,receives%2010%20calls%20per%20hour.) ($\star$)

- 콜센타에 걸려오는 전화의 수, 1시간동안 
- 레스토랑에 방문하는 손님의 수, 하루동안 
- 웹사이트를 방문하는 사람의 수, 1시간동안 
- 파산하는 사람의 수, 1달동안 
- 네트워크의 끊김 수, 1주일동안 
"""

# ╔═╡ a53f8d2c-7596-43a8-96b8-4ad8bf6de374
md"""
맥도날드에 n 시간동안 사람이 몇명이나 들락날락하는가?
"""

# ╔═╡ cc97ee44-c237-46bc-8a51-a7f288ce3787
md"""
!!! note "포아송분포의 느낌"
	단위시간 (혹은 단위공간) 에서 발생하는 어떠한 이벤트 수를 X라고 하면 X는 포아송분포를 따름. 
"""

# ╔═╡ 11b8b1d0-4a93-4c16-b961-fc9c85742a14
md"""
### B. 모수 $\to$ 히스토그램
"""

# ╔═╡ 4ef09487-f353-4bf8-9add-a221e4801098
md"λ = $(@bind λ Slider(0.1:0.1:30, show_value=true, default=1))"
#λ = @bind λ Slider(0.1:0.1:30, show_value=true, default=1)

# ╔═╡ fda3dd70-f632-4958-999d-d736ff52b1dd
let
	N = 100
	histogram(rand(Poisson(λ),N))
	xlims!(0,60)
end

# ╔═╡ bbc73fe7-ba62-4b53-95e9-adb9e3c09e69
md"""
### C. 난수생성 테크닉
"""

# ╔═╡ 11090108-2138-4ec7-bbdb-5da6a8e7ab2e
md"""
-- 방법1 : 이항분포의 제약된 극한이 포아송 분포임을 이용..!
"""

# ╔═╡ 6c30a97e-34a1-488b-9da6-0f71e5b1e71b
rand(Poisson(3),10)

# ╔═╡ 0c6f0e07-b19f-4020-a3ab-e16bc3aaa68a
md"""
-- 방법2: 이항분포의 포아송근사
"""

# ╔═╡ 3e4b7b2b-dbab-4a34-81b7-cdb15d15a21a
md"""
!!! note "이론: 이항분포의 포아송근사" 
	이항분포에서 (1) $n\to \infty$ (2) $p\to 0$ 이면 이것은 평균이 $\lambda=np$ 인 포아송분포와 비슷해진다. 즉 평균이 $\lambda$인 포아송분포는 $B(n,\frac{\lambda}{n})$로 대신 만들 수 있다. 
"""

# ╔═╡ 45841bb0-8881-48af-baae-ea5555a0482e
let 
	N = 1000
	λ = 5
	n = 10000 #n이 작으면 별로 안비슷한데,,, 커질수록 비슷해지는 것을 볼 수 있다.
	p = λ/n
	X = rand(Binomial(n,p),N)
	Y = rand(Poisson(λ),N)
	@show (n,p), λ #밑에 보이는 bar가 이 코드를 통해 생성된 것.
	#--#
	p1= histogram(X); xlims!(0,50); title!("(a) (포아송처럼 보일순 있지만) 이항분포")
	p2= histogram(Y); xlims!(0,50); title!("(b) 포아송")
	plot(p1,p2,layout=(2,1)) # 배치 구도 정렬
end

# ╔═╡ 60321f75-4f38-4ef0-ba6a-023ead960ea4
md"""
-- 방법3: 베르누이 $\to$ 이항분포 $\approx$ 포아송
"""

# ╔═╡ f415c9dd-b493-4f29-a222-84c8c9e5f5dd
md"""
사실: 전북대 맥도날드에는 항상 1분에 평균 6명의 손님이 방문한다. (느낌? 평균이 6인 포아송) (포아송 프로세스에 대한 설명..)
- 그럼 10초에는 대충 1명의 손님이 오지 않겠어?
- 그럼 1초에는 대충 0.1명의 손님이 오지 않겠어?
- 그럼 0.1초에는 대충 0.01명의 손님이 오지 않겠어?

생각: (x,x+0.1초) 에서 방문객의 분포와 (x+0.1초,x+0.2초) 방문객의 분포는 독립일까? 분포는 다를까? 
- 딱봐도 분포는 같고, 독립이어보임. 

아주 짧은 시간을 설정함으로써 아주 짧은 시간동안의 베르누이 시행으로 보겠다는 아이디어... 한번의 그 짧은 시간동안 2번의 사건이 발생하면 안된다.

주장: 
- 주장1: 0.1초동안 맥도날드에 오는 손님수 X는 $X \sim Ber(0.01)$ 라고 봐도 거의 무방. 
- 주장2: 그렇다면 60초동안 맥도날드에 오는 손님수 X는 $X \sim B(600,0.01)$ 이어야 한다.
- 주장3: 그런데 $B(600,0.01)$은 충분히 큰 $n$과 충분히 작은 $p$를 가지고 있다. 따라서 이것은 $Poi(6)$의 분포와 비슷할 것이다. 
"""

# ╔═╡ bb216082-b7ef-4620-b7b0-3670da232ef8
let
	histogram(rand(Poisson(6),1000))
	[sum(rand(600) .< 0.01) for i in 1:1000] |> histogram!
end

# ╔═╡ 813dd023-684f-49fe-a202-c01c58fe2530
let 
	N = 1000
	n = 6000
	p = 0.001
	λ = n*p
	X = [rand(Bernoulli(p),n) for i in 1:N] .|> sum
	Y = rand(Poisson(λ),N)
	@show (n,p), λ
	#--#
	p1= histogram(X); xlims!(0,50); title!("(a) 베르누이 -> 이항분포 ≈ 포아송")
	p2= histogram(Y); xlims!(0,50); title!("(b) 포아송")
	plot(p1,p2,layout=(2,1))	
end

# ╔═╡ cdcf3ac3-4873-424e-a035-d3c5e7857e7f
md""" 
!!! note "포아송 프로세스 느낌"
	하여튼 (1) **"엄청 짧은 시간"**에 (2) **"엄청 작은 확률"**의 베르누이 시행이 (3) **"엄청 많이 독립적으로 반복"**되는 느낌을 꼭 기억하세요!
"""

# ╔═╡ 85e113f2-625c-4d57-b3fe-1167b9f3c5ca
md"""
### D. 분산이 특이하네? 
"""

# ╔═╡ 914b8c7f-e444-4fa9-bc5f-27228a76193a
md"""
-- 떡밥..
"""

# ╔═╡ 58ab80d4-f59d-4471-9a3f-3153873e9107
md"""
### E. 포아송 특징
"""

# ╔═╡ 54703f3f-0ed5-4909-b24b-2701f6f904fb
md"""
-- 포아송분포의 합은 다시 포아송분포가 된다. 
"""

# ╔═╡ 794cbef2-e38e-456b-8b61-0d270f764a33
md"""
!!! note "이론: 포아송분포의 합"
	포아송분포의 합은 다시 포아송분포가 된다. 
	$X \sim Poi(\lambda_1), Y\sim Poi(\lambda_2),~ X \perp Y \Rightarrow X + Y \sim Poi(\lambda_1 + \lambda_2)$ 
"""

# ╔═╡ a6f03cd0-4eae-4521-a7d3-cd1a1ba61f4d
md"""
의미? (1) 1분동안 맥도날드 매장에 들어오는 남자의 수는 평균이 5인 포아송 분포를 따름 (2) 1분동안 맥도날드 매장에 들어오는 여자의 수는 평균이 4.5인 포아송분포를 따름 (3) 남자와 여자가 매장에 오는 사건은 독립 => 1분동안 맥도날드 매장에 오는 사람은 평균이 9.5인 포아송 분포를 따른다는 의미. 
"""

# ╔═╡ 5a3cd616-68dd-41ba-a3a8-7f1199c51707
md"""
-- 실습
"""

# ╔═╡ 91ce515f-b508-462f-9d3a-5a14d634b41d
let 
	N= 10000
	X = rand(Poisson(5),N) # 남자
	Y = rand(Poisson(4.5),N) # 여자
	p1 = X.+Y |> histogram ; title!("(a) Poi(5)+Poi(4.5)") ; xlims!(0,25)
	p2 = rand(Poisson(9.5),N) |> histogram ; title!("(b) Poi(9.5)") ; xlims!(0,25)
	plot(p1,p2,layout=(2,1))
end

# ╔═╡ bde63d4b-aee5-41a1-9dde-f2777ff7879d
md"""
## 6. 숙제
"""

# ╔═╡ 35749112-78e1-47da-b409-80374e4cd327
md"""
`1`. 수업시간에 소개한 이항분포를 만드는 3가지 방법으로 (n,p)=(30,0.45)인 이항분포 100를 만들라. 세 방법의 히스토그램을 비교해보라.

`2`. "균등분포 $\to$ 베르누이 $\to$ 이항분포 $\approx$ 포아송" 의 방법으로 $Poi(12)$의 분포를 근사하고 히스토그램을 비교해보라. 
"""

# ╔═╡ 6c91cd83-5d8b-4b95-a121-a04039d19256
md"""
≤ #\le lower or equal 연산자.
"""

# ╔═╡ ead24617-fd8f-43dc-96a7-c8b128b5b6f3
let
	N = 10000
	λ = 12
	n,p = 10000,0.0012

	unif = [sum(rand(n) .<p) for i in 1:N]
	poi = rand(Poisson(λ),N)

	histogram(unif)
	histogram!(poi)
end

# ╔═╡ b927a273-ac24-4e94-bcd7-7857f2e3c8ab
##1
let
	#method1
	p1 = histogram([rand(30) .< 0.45 for i in 1:1000]  .|> sum,color = 1, label = "uniform만 사용");title!("bia Uniform");xlims!(0,25)
	#method2
	p2 = histogram([rand(Bernoulli(0.45),30) for i in 1:1000] .|> sum ,color = 2, label = "Bernoulli 사용");title!("bia Bernoulli");xlims!(0,25)
	#베르누이 난수를 이용한 이항분포 난수 생성
	#method3
	p3 = histogram(rand(Binomial(30,0.45),1000),color = 3, label = "Binomial 사용");title!("bia Binomial");xlims!(0,25)

	plot(p1,p2,p3 , lay_out = (3,1))
end

# ╔═╡ a5c0f298-d209-4088-8bfd-439528c87f71
##2
let
	N = 1000
	n,p = 12000,0.001
	λ = n*p 
	
	P = rand(Poisson(λ),N)
	
	p1 = histogram([rand(12000) .< 0.001 for i in 1:1000] .|> sum,color = 1, label = "Unif -> Bernoulli -> Binomial" );title!("Method 1")
	p2 = histogram(P, label = "Poisson",color= 2);title!("Method 2")

	plot(p1,p2 , layout = (2,1))
end	

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.25.107"
Plots = "~1.39.0"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "e29a6a28cc9520ca196d95e1c50bce5da36039f6"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a4c43f59baa34011e303e76f5c8c91bf58415aaf"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "0f4b5d62a88d8f59003e43c25a8a90de9eb76317"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.18"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "bfe82a708416cf00b73a3198db0859c82f741558"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.10.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "359a1ba2e320790ddbe4ee8b4d54a305c0ea2aff"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.0+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "8e59b47b9dc525b70550ca082ce85bcd7f5477cd"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.5"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cad560042a7cc108f5a4c24ea1431a9221f22c1b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.2"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dae976433497a2f841baadea93d27e68f1a12a97"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.39.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0a04a1318df1bf510beb2562cf90fb0c386f58c4"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.39.3+1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "af81a32750ebc831ee28bdaaba6e1067decef51e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3da7367955dcc5c54c1ba4d402ccdc09a1a3e046"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "532e22cf7be8462035d092ff21fada7527e2c488"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.6+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─05e369d7-e0b8-4ef4-8831-938423a6717e
# ╟─db4dd854-1764-4a62-a77d-498ac1c008b1
# ╠═3e2405d2-5c92-42cd-bbc6-66c1add0d282
# ╟─77c4a801-d098-4f45-9b99-4e7e6f55a10b
# ╠═3e7b4e0d-c52e-4214-a8fb-7ae7b03a60e9
# ╠═5bf2d19f-b8bb-43be-bc35-a963a01f71cb
# ╠═6155f666-e228-463e-9721-39ee899366bf
# ╟─f94127c6-ca98-4946-babf-cb08768a7b4b
# ╟─74dbc7ae-3293-495b-bb3c-e633bfa76c33
# ╟─68ce3dd0-fa08-468d-bd54-1f2f383faab6
# ╟─5d2716e2-c5d3-470e-8178-500f313172c4
# ╠═239478fc-539d-48de-aba8-73dafcb4a6c9
# ╠═00dbac24-e344-4d79-9b5c-5cf7541d33bb
# ╠═3ca030b1-4de8-4460-a16d-4aae3627c82a
# ╠═14e43717-d59d-4184-b40b-df6bd0fb94b0
# ╠═8537c3e7-d2f6-41ce-8563-3136317c16d6
# ╠═5574fb2e-f9a1-47ca-bd19-6ae28a2d69b5
# ╟─d39cc97b-f7ad-4e54-a6ed-0497d6346acf
# ╟─8dd2007b-dfc3-453a-96c9-7e1df21c871e
# ╟─4beb013b-6540-48b2-9d52-818bf34f7ef1
# ╠═710c494e-41cd-4580-ab7d-748516e894bf
# ╠═eec1ffe4-13da-49d1-8af6-778a543a6bdd
# ╠═15e246db-44ae-48ac-bfa1-a27fca6cb5e2
# ╟─69cbd1cc-8042-4921-bc49-74420f7f3ae1
# ╠═a0c5fa5c-4fef-4862-9ce2-895c03eba9d5
# ╠═ae9b80c4-eee7-4032-aa5e-ab195fb75722
# ╟─a302696f-3667-488c-a2c0-a95eca178801
# ╟─83966cac-2ba7-4751-a571-e0c36840731a
# ╟─b1eb6071-3da2-44ce-bc8e-7ad7f5910bb3
# ╟─caa9606b-a2b9-4d2b-93c5-453f4525b051
# ╟─a5013797-d2b8-4f53-8f30-0733f248e9b8
# ╠═b402c405-c6f6-4f53-886b-05b12021b6c3
# ╟─bba8738c-2d5c-4d40-bb3d-1a11a57fd9d6
# ╠═74ab3bea-c2d2-4fe3-acd7-7de9b007b7ef
# ╠═59de810d-989a-4dca-99d5-2045d8785cfc
# ╠═435bbf1e-531b-4a0e-9b69-3b756e2371b3
# ╟─27f8e253-01e9-44e6-bb7d-c1de2a8d37c9
# ╠═d0d7b51a-9edf-426b-b3e4-cded5f91e6c7
# ╟─50a1bd66-19eb-41e4-a0e5-4d7369112a4d
# ╟─1a2e694b-322d-48e4-b381-ab2cd277366f
# ╠═b4b0a042-6936-4dc0-b212-b45b7131d495
# ╠═de89581f-47fd-48e9-a6d9-d22311700e83
# ╟─78f8e0a8-e1aa-4c70-a460-a179456bd8fc
# ╟─82b80b24-6fb2-43c4-a72c-d0c7b46c403c
# ╟─c147f8a3-43b7-447c-9d15-fc6635af4221
# ╟─d74a7832-9481-44fc-9e93-7c4e6dda5664
# ╠═886524f9-447a-4661-adb3-f345314d4750
# ╠═3009eb43-63ce-46ec-ad4e-c20acee8e261
# ╟─9e1dd3ca-7a75-4db8-bed9-8885d0678ad6
# ╠═8dde41a2-7164-4568-9cbe-c91f4e29bf5f
# ╠═8ea5c74c-95a8-4980-a144-3f2cf1fb0453
# ╠═f5627b38-1bee-4c03-b83c-79b2632d19bd
# ╟─33d3125e-3d5f-467c-8488-2e82f9832e51
# ╟─631466f1-e3a8-4df1-ad1e-255a3b1e6e32
# ╠═d8bb13c9-9ad9-473d-87ed-dacbefc23cf2
# ╠═e2abb175-945b-48a7-a806-a05606e92290
# ╟─69dd4150-f82a-40a7-a1e1-ab2c90f024dd
# ╠═682bf567-0e62-475c-9078-6d873b6ac911
# ╟─4b7789e6-ff5f-4000-b8ff-7d888d4e2e97
# ╠═a2ca922b-432a-4215-b51b-32db273852d1
# ╠═1c838c8b-cb51-4d7b-87ab-95cb3780ad2b
# ╠═aa72e48f-9b28-46d8-9cbc-7cd0c1191879
# ╠═e86e0e87-313a-4851-8cc3-00011d7cc252
# ╟─56c84dde-be00-48c4-a074-6f90c39e7092
# ╠═3d63d57c-a54f-4571-a514-c686d6415b70
# ╠═ce2c1125-8114-4b74-a86a-2dacffd4a811
# ╠═12d5fd97-7a1c-405f-87d6-fa5317636852
# ╠═8cdee83a-d7bf-4c75-a7c5-35168b474ffc
# ╠═87fcf7c3-0659-486f-9924-2978e2fe45b2
# ╟─7a474be8-a3f0-4b7f-94ab-90eda88bfefc
# ╟─d24f2b08-8997-4cc5-8f87-4a5cbd8290f2
# ╠═7e1e6e3f-f13a-414d-9b12-b1a610f61444
# ╟─6c429ac2-2cca-45a5-a3dc-e09f94275732
# ╟─883ceb79-7e3d-496a-b76c-78d0e586c1d9
# ╟─fb89feac-7ee3-424c-8367-8f4ef577b12c
# ╟─fc2b7c5b-9cb7-4536-8f55-7a502319b3c6
# ╠═8f4f4c31-b997-4bf7-b8ee-77f5c1c90b75
# ╟─3b13a7d5-d476-4d97-bfef-374bbf136b53
# ╟─a53f8d2c-7596-43a8-96b8-4ad8bf6de374
# ╟─cc97ee44-c237-46bc-8a51-a7f288ce3787
# ╟─11b8b1d0-4a93-4c16-b961-fc9c85742a14
# ╠═4ef09487-f353-4bf8-9add-a221e4801098
# ╠═fda3dd70-f632-4958-999d-d736ff52b1dd
# ╟─bbc73fe7-ba62-4b53-95e9-adb9e3c09e69
# ╟─11090108-2138-4ec7-bbdb-5da6a8e7ab2e
# ╠═6c30a97e-34a1-488b-9da6-0f71e5b1e71b
# ╟─0c6f0e07-b19f-4020-a3ab-e16bc3aaa68a
# ╠═3e4b7b2b-dbab-4a34-81b7-cdb15d15a21a
# ╠═45841bb0-8881-48af-baae-ea5555a0482e
# ╟─60321f75-4f38-4ef0-ba6a-023ead960ea4
# ╟─f415c9dd-b493-4f29-a222-84c8c9e5f5dd
# ╠═bb216082-b7ef-4620-b7b0-3670da232ef8
# ╠═813dd023-684f-49fe-a202-c01c58fe2530
# ╟─cdcf3ac3-4873-424e-a035-d3c5e7857e7f
# ╟─85e113f2-625c-4d57-b3fe-1167b9f3c5ca
# ╟─914b8c7f-e444-4fa9-bc5f-27228a76193a
# ╟─58ab80d4-f59d-4471-9a3f-3153873e9107
# ╟─54703f3f-0ed5-4909-b24b-2701f6f904fb
# ╟─794cbef2-e38e-456b-8b61-0d270f764a33
# ╟─a6f03cd0-4eae-4521-a7d3-cd1a1ba61f4d
# ╟─5a3cd616-68dd-41ba-a3a8-7f1199c51707
# ╠═91ce515f-b508-462f-9d3a-5a14d634b41d
# ╟─bde63d4b-aee5-41a1-9dde-f2777ff7879d
# ╟─35749112-78e1-47da-b409-80374e4cd327
# ╟─6c91cd83-5d8b-4b95-a121-a04039d19256
# ╠═ead24617-fd8f-43dc-96a7-c8b128b5b6f3
# ╠═b927a273-ac24-4e94-bcd7-7857f2e3c8ab
# ╠═a5c0f298-d209-4088-8bfd-439528c87f71
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
