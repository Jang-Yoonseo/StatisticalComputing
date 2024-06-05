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

# ╔═╡ 9b4ce6c2-da7b-11ee-251b-8bc123c638b2
using PlutoUI, Plots

# ╔═╡ bab377cb-3b33-4fb2-a040-22ba7cb9c033
md"""
# 1. 과제 캡쳐본 by 장윤서
"""

# ╔═╡ d74dcf5a-3154-4c13-89e0-ab5b04eb422f
md"""
# 2. Imports
"""

# ╔═╡ 2d5aa4fd-334d-4b67-9447-20d71dda4d89
PlutoUI.TableOfContents()

# ╔═╡ 34605b2e-b908-4575-ab50-b548327edb05
Plots.plotly() #interactive하게 해줌.

# ╔═╡ 5f451d66-392c-4843-b725-1d09a2fe35fd
md"""
# 3. 줄리아 vs (R,Python)
"""

# ╔═╡ fc4555e6-3d64-4393-b782-ad543b9ff0ed
md"""
## A. 신기한 변수이름
"""

# ╔═╡ b8ab403c-9bc3-4467-a333-a0f84d7c5d92
md"""
-- 일반적인 변수이름
"""

# ╔═╡ 50c3f495-5570-4eda-9c4b-663db29fa992
n = 100

# ╔═╡ b4948c5a-45a0-4cc6-9037-605fddd22c51
md"""
-- 신기한 변수이름들
"""

# ╔═╡ a3eb5452-abf0-4b05-8a6f-f9e53166ce7c
δ = 0.1 

# ╔═╡ 113f26a7-abd4-46bd-b3d2-fc2aeef12328
π # 이미 저장되어있음. 

# ╔═╡ 8305b24d-3d6f-421d-832e-7ff638a76b20
ℯ # 이미 저장되어있음

# ╔═╡ 192be54a-b99d-4390-808f-b5015aa2eb1d
x̄ = (1+2+3)/3 # x\bar
#인터렉티브 하기 때문에 하나를 시행하면 다른 하나가 시행 불가!

# ╔═╡ cc3ac75c-8153-4b62-95a3-68e3bf3fa6a0
β̂ = 2.4 #\beta\hat

# ╔═╡ 317b2432-73a8-451c-b784-6c7c16dcc5b7
x₁ = 1 #x\_1 아래 첨자..

# ╔═╡ a053cb96-3379-4566-a4ab-fc71c6bda6e8
md"""
## B. 신기한 연산자
"""

# ╔═╡ aa4dcb86-1442-43ad-9ba5-5be64da404df
md"""
-- `≈` 연산자
"""

# ╔═╡ 16d79530-6f81-46a1-b3f3-4973061befad
md"\approx 를 입력하면 근사 연산자가 나온다."

# ╔═╡ 6a656877-944d-49cf-8367-2627cb21a1a4
1 ≈ 1.000000000000001 #근사

# ╔═╡ fe6488f2-7bda-4ae9-b5e1-4bc6016814ae
1 ≈ 1.00000001 #여기서 부터 거의 같다고 본다..

# ╔═╡ 677459ab-bf9c-458c-8a02-7d8b14217ae9
1 == 1.000000000000001 #Exactly same 한가?

# ╔═╡ 8a7f4fa1-e61a-4164-9f67-a4a1f17700e8
md"""
-- 비교연산자
"""

# ╔═╡ 99516c11-06ae-4cdc-8c38-68cb4d0367d6
1 ≤ 1.0  #\le (less of equal 연산자)
#≥ #이건 \ge greater of equal 인듯

# ╔═╡ 53dc36c4-0cc7-4d9a-b72d-bbd0d6296682
md"""
## C. 편리한 함수사용
"""

# ╔═╡ 0a2cb594-5c1a-4c5d-9fe4-2501f888e974
md"""
-- 일반함수
"""

# ╔═╡ 1dd1961b-20fc-4805-9fd9-18e743036b20
f(x) = x+5 #x라는 입력값을 받는 함수 오브젝트 f 생성

# ╔═╡ bc73915a-b65c-4a3a-83a9-4b4b1d9b6b3f
f(2)

# ╔═╡ 59177c93-6ed8-4f1b-9047-7dacbd0d0fc2
md"""
-- 함성함수
"""

# ╔═╡ b21a3c67-8854-4f4a-bc7f-c611127151f7
g(x) = 2x

# ╔═╡ b6f57929-e80a-49df-a5da-3688b2bdfd97
g(4) #우리가 아는 그 행렬처럼 작동~

# ╔═╡ 4d8b1cbc-7147-400d-b922-8cb99d15c4cf
(g∘f)(1)  #\circ

# ╔═╡ e0b880d3-c755-4cf2-9d83-efb05dc7de0d
md"""
`-` 파이프연산자
"""

# ╔═╡ b996020b-b910-42d5-adf8-586678a28615
1 |> f |> g #엘레멘트 와이즈 하게 하려면 앞에 점 붙이기~

# ╔═╡ f3f605ef-ecc1-43d7-8570-746b6196b34b
md"""
## D. 편리한 매트릭스 선언
"""

# ╔═╡ dd982018-c9dc-48e3-88ca-0215c520e930
X = [1 2 
	 3 4]

# ╔═╡ f2b83b10-5b90-4ffa-8b1c-07ba797b2d3b
[1 3 4
5 6 7]  #행렬 생성하기

# ╔═╡ b25d55fc-06ca-43a8-ad09-0d37dc4e922e
Y = [1 3 4
	 5 6 7 8] #이렇게 하면 당연히 안됩니다.

# ╔═╡ abf14fd7-1d6f-4d23-90d2-c2c30f0754cd
md"""
## E. 간결한 연산
"""

# ╔═╡ a8a79e4c-e3f0-4147-80ce-732bc8c185f6
md"""
-- 예시1
"""

# ╔═╡ 8b94d99c-d17f-442d-ac48-544ff11636b0
2(1+2) #곱셈기호 안넣어도 알아서 잘 해준다.

# ╔═╡ d99a3632-59bb-47fd-a2fd-165494f37985
2*(1+2) #넣어도 당연히 잘 해줌

# ╔═╡ 309e43e4-26b3-4370-958f-e776e2364031
md"""
-- 예시2
"""

# ╔═╡ a4ece2bd-9c80-4654-abe1-563da8f5d78b
X'X

# ╔═╡ 8b920d78-0dbe-4cff-83be-4c1abda9e92c
md"""
-- 주의: 의미가 명확한 경우만 생략가능
"""

# ╔═╡ 7454e22a-8e92-4136-804b-02e8f507420c
XX' #XX 라는 오브젝트의 transpose라고 해석해버림

# ╔═╡ ee64f7a5-4d23-4a06-a9b6-11043b4b178f
md"""
## F. 인덱스
"""

# ╔═╡ 14aa4bf5-9f01-4e79-b3f0-e49daeae12da
lst = [11,22,33]

# ╔═╡ 05be06e6-5bb7-4c31-acb7-bab97652b557
lst[1] #이건 R과 유사.. 인덱스가 0부터 시작하지 않아요

# ╔═╡ 14f0f686-1a41-4eb8-9354-e004505b34b2
md"""
## G. 컴프리헨션
"""

# ╔═╡ defe15c2-7153-46e9-9ece-4a9be738be3b
[x^2 for x in [1,2,3]] #리스트 컴프리헨션.. 파이썬과 유사

# ╔═╡ 976688f3-5748-4435-b842-7f402362babc
md"""
## H. 튜플
"""

# ╔═╡ 83c56718-06b7-4953-b831-c02c069cf890
md"""
-- 언패킹
"""

# ╔═╡ ef1acd1d-4ddd-4532-97a1-b41bba8a7b75
x1,x2,x3 = 1,2,3

# ╔═╡ 7f8243ce-dd87-4214-8122-cbab194e9c30
md"""
-- 다중출력
"""

# ╔═╡ 7f6ed435-1972-419c-a27c-0bd69a47faa9
x1,x2

# ╔═╡ ddf7bfe6-43bd-4acb-aab8-3c91aaa7cde6
x1,x3

# ╔═╡ 84086422-923c-4235-aedf-3f8de6683ee5
x1

# ╔═╡ 35ac1d21-1026-432a-bd49-91efd71ec6c6
md"""
# 4. 플루토 vs 주피터
"""

# ╔═╡ 920d4a27-0b5c-487e-9e43-32f3ec26c4a1
md"""
## A. 변수이름이 중복될 경우
"""

# ╔═╡ 9922b0a6-dad2-40ba-beb5-235d32d08086
# ╠═╡ disabled = true
#=╠═╡
a=1
  ╠═╡ =#

# ╔═╡ f97e3e4d-99c7-445c-8a58-18d15f5e3d2d
a = 2 #변수 이름이 중복될 경우,, 상단에 있던 변수가 무효화 됨..
#인터렉티브 하기 때문~~

# ╔═╡ a455455d-97a1-4583-85a3-4d2599ace0ef
md"""
## B. 인터렉티브 노트북
"""

# ╔═╡ edc51a4b-0081-4237-bc0a-56abf7252efc
x = 2 #여기 숫자를 바꾸거나

# ╔═╡ bb8bc581-bd2b-4d45-95fd-9f7be09b0383
y = 5 #여기 숫자를 바꾸면 아래 결과가 계속 달라짐.

# ╔═╡ 8af35e77-1ea0-4f79-97d9-11749fdd8765
z = x + y

# ╔═╡ 44e0cc5f-a3d6-4b19-8621-ec1dcec4f8c8
md"""
## C. 여러줄의 코드 -- 단점같음
"""

# ╔═╡ 9a33e66c-0831-43de-bb10-c8324b62a95a
md"""
-- 예제1: 줄리아에서는 한 셀에 여러줄의 코드를 작성할 수 없다. 
"""

# ╔═╡ c0e51846-4c1f-45db-a930-6a3c93b9c742
k=1
k=k+1

# ╔═╡ 50d8de79-d55a-4958-b310-28752f11665d
md"""
-- 예제2: 줄리아에서 한 셀에 여러줄의 코드를 작성하고 싶다면 아래와 같이 begin - end 를 사용한다.
"""

# ╔═╡ 159827aa-aa38-4f3d-b1f3-8e138d72cde4
begin #전역적으로 작동
	k=4
	k=k+1
end

# ╔═╡ d666c6ce-9831-4902-81d0-29de1c6c0c2e
k

# ╔═╡ 79df769a-2b67-49d6-86a8-b1220e7dd993
md"""
-- 예제3: 줄리아에서 한 셀에 여러줄의 코드를 작성하는 다른 방법은 let - end 를 사용하는 것이다.
"""

# ╔═╡ 29a5884f-5e14-4371-8c73-26c67e248aad
md"begin - end 랑 let - end는 차이가 있는 듯.."

# ╔═╡ fb3175b4-63f1-4723-8381-fa4dd46245b0
let # 이 cell에서만 l을 사용하고 전역적으로는 사용하지 않음..
	l=1 
	l=l+1
end

# ╔═╡ a77e4af8-1d18-4391-b509-7267d23e3bfe
l #Global하지 않다..

# ╔═╡ eb1d514a-bfae-4132-adf5-a4fba73f63e0
md"""
## D. 마크다운 셀
"""

# ╔═╡ f5761099-b753-48a1-ac93-d3c1ef5ee0f1
md"""
-- 마크다운셀이 따로 있는게 아니고 `md"..."`를 이용하여 사용하면 된다.
"""

# ╔═╡ 6d068573-9f54-4c46-8c5b-08ca7f93a551
md"줄리아프로그래밍: 담당교수 최규빈"

# ╔═╡ 0d0a3c5f-d891-4983-a120-1950f8ce45af
md"""
-- `$()`를 이용하여 변수를 끼워넣을 수 있다. (마치 파이썬에서 f string처럼!!)
"""

# ╔═╡ 7b534287-d060-4216-968f-181eeb68ded6
name = "최규빈 교수님"

# ╔═╡ 16bd1844-f8b3-4811-9c96-18d481593bff
md"줄리아프로그래밍: 담당교수 $(name)" #끼워넣기??!!

# ╔═╡ f6c3d52c-f7a4-4e3a-a0e9-db0740187938
md"$x^2$" #그라하하!

# ╔═╡ 116480c6-33d6-458e-b589-ea824ffe1ea7
md"""
## E. 인터렉티브 플랏
"""

# ╔═╡ df72114f-6e2e-4af0-b80d-d443b8e7c560
scatter([1,2,3,4],[1,2,4,3]) # interactive plot이 되게끔 설정을 했기 때문.

# ╔═╡ 06a20ad0-24bc-4f6d-b7ec-d22df054c28f
md"""
# 5. 위젯
"""

# ╔═╡ 62997e2b-3177-4949-97d2-19f356829171
md"""
## A. 슬라이더
"""

# ╔═╡ 5fd8e8c7-ad55-4dce-832f-0c49a74d691c
md"""
-- 슬라이더를 이용하여 $\alpha$ 값을 설정.
"""

# ╔═╡ 439b946f-188d-48e9-aa30-3adacd6b52f5
k1 = @bind kk Slider(-1:0.1:2, show_value = true) #이러면 그냥 슬라이더가 만들어짐.

# ╔═╡ c6f9c07f-7a7a-4103-b699-623a6108906f
kk #슬라이더 객체의 숫자가 kk라는 객체에 저장되었다.

# ╔═╡ 6f0e4b4e-d8bc-48b1-8c86-4e4f103e05ca
k1 #은 그냥 슬라이더 객체 그자체

# ╔═╡ 6b96702c-c3f4-41bb-8a4a-cd227db71a92
md"kk =  $@bind k2 Slider(-1:0.1:2, show_value = true,default = 1.1)" #이것도 되고
#md"kk =  $(@bind k2 Slider(-1:0.1:2, show_value = true))" #이것도 되고
#md"kk =  $@(bind k2 Slider(-1:0.1:2, show_value = true))" #이것도 됨..

# ╔═╡ 8064b33b-c00b-4dd6-9fe0-a2d7bc14132a
k2

# ╔═╡ bb140735-a485-41ef-ac6b-f8fbfa429d9e
md"#### 구분을 잘 해보도록,,,"

# ╔═╡ e06912a1-14e3-4664-9967-4a6f624a3c32
@bind α Slider(-1:0.1:1)

# ╔═╡ f1c32574-dd94-425c-890a-7692524ab5f7
α

# ╔═╡ 07a46c0f-bb89-446e-a6ad-f78efa078a1b
md"""
-- 숫자가 옆에 보이는 슬라이더
"""

# ╔═╡ 982537c6-7189-403c-aca4-819e93604747
@bind β Slider(-1:0.1:1,show_value=true)

# ╔═╡ f34199b8-e39e-44cd-85e7-0fb362574b21
#실습
@bind η Slider(-1:0.1:1, show_value=true)

# ╔═╡ 30905821-8eb2-476f-bd32-65f7ee9a6fcc
md"""
-- 슬라이더를 오브젝트로 받을수도 있음
"""

# ╔═╡ 7fec9962-bd09-4a17-8602-7a30c35f8aa3
gamma_selector = @bind γ Slider(-1:0.1:1,show_value=true)

# ╔═╡ 2f13d9e1-70c5-4b49-8e7a-82298640d2a3
gamma_selector #슬라이더 객체 그잡채

# ╔═╡ aeadfda5-b3e5-486e-a255-73b1174d10c0
γ #슬라이더에 바인딩 된 값

# ╔═╡ 0bef159a-69fa-4af9-8fe5-8ee4977e02f0
md"""
-- 응용: $f(x)=(x-\alpha)(x-\beta)(x-γ)=0$ 의 그래프
"""

# ╔═╡ 2e8c063e-a9cc-49c4-b428-ee32de46ddd9
let 
	f(x) = (x-α)*(x-β)*(x-γ)
	plot(-1:0.01:1,f)
end 

# ╔═╡ 39cbd840-3821-4255-bccc-a837fcbe3c4f
md"""
## B. 라디오버튼
"""

# ╔═╡ 90956254-e8f6-4284-9691-652311dc4f2f
md"""
-- 라디오버튼 사용: @bind vote Radio(리스트)
"""

# ╔═╡ c17da9ad-b9ed-46c5-a74e-1ad5466a27f9
@bind vote Radio(["찍먹","부먹"]) #위와 비슷하곘지? 찍먹 부먹을 고르면 vote라는 객체에 저장된다..

# ╔═╡ 28a73360-59f0-4ff6-a5bc-b51dfd7e2c1a
vote

# ╔═╡ b7b53e73-66b7-494a-a376-e00244d1dbb8
md"""
-- 선택결과를 `md"..."`로 출력
"""

# ╔═╡ e11e00a1-2b3d-453a-8fe7-d2127fd67838
#md"저는 "$vote"파 입니다" #이렇게 하면 따옴표 때문에 안됨.
md""" 저는 "$vote" 파 입니다 """

# ╔═╡ 670a27bb-0c36-4a19-9b9d-0c92e4f9df19
# 실습
@bind toopyo Radio(["윤석열","이재명"])

# ╔═╡ 6ca53978-fa5c-4ec9-93a0-eafd353fe793
toopyo

# ╔═╡ 4548b4ae-1234-4e2e-9171-8d0c8eab64b8
# 실습
objct = @bind toopyo1 Radio(["윤석열","이재명"]) #라디오버튼 객체 그자체

# ╔═╡ 6adc6c7b-df5e-4147-92df-68a147ba7430
objct

# ╔═╡ 3d263527-0937-4ecf-9545-611abcda1021
md"같은 오브젝트를 호출한 후 하나를 수정하면 따라서 수정된다.."

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Plots = "~1.39.0"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "52934e2881e5b365a298a28396240865a5fc2a3c"

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
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

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
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

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

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

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
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

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
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

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
# ╟─bab377cb-3b33-4fb2-a040-22ba7cb9c033
# ╟─d74dcf5a-3154-4c13-89e0-ab5b04eb422f
# ╠═9b4ce6c2-da7b-11ee-251b-8bc123c638b2
# ╠═2d5aa4fd-334d-4b67-9447-20d71dda4d89
# ╠═34605b2e-b908-4575-ab50-b548327edb05
# ╟─5f451d66-392c-4843-b725-1d09a2fe35fd
# ╟─fc4555e6-3d64-4393-b782-ad543b9ff0ed
# ╟─b8ab403c-9bc3-4467-a333-a0f84d7c5d92
# ╠═50c3f495-5570-4eda-9c4b-663db29fa992
# ╟─b4948c5a-45a0-4cc6-9037-605fddd22c51
# ╠═a3eb5452-abf0-4b05-8a6f-f9e53166ce7c
# ╠═113f26a7-abd4-46bd-b3d2-fc2aeef12328
# ╠═8305b24d-3d6f-421d-832e-7ff638a76b20
# ╠═192be54a-b99d-4390-808f-b5015aa2eb1d
# ╠═cc3ac75c-8153-4b62-95a3-68e3bf3fa6a0
# ╠═317b2432-73a8-451c-b784-6c7c16dcc5b7
# ╟─a053cb96-3379-4566-a4ab-fc71c6bda6e8
# ╟─aa4dcb86-1442-43ad-9ba5-5be64da404df
# ╟─16d79530-6f81-46a1-b3f3-4973061befad
# ╠═6a656877-944d-49cf-8367-2627cb21a1a4
# ╠═fe6488f2-7bda-4ae9-b5e1-4bc6016814ae
# ╠═677459ab-bf9c-458c-8a02-7d8b14217ae9
# ╟─8a7f4fa1-e61a-4164-9f67-a4a1f17700e8
# ╠═99516c11-06ae-4cdc-8c38-68cb4d0367d6
# ╟─53dc36c4-0cc7-4d9a-b72d-bbd0d6296682
# ╟─0a2cb594-5c1a-4c5d-9fe4-2501f888e974
# ╠═1dd1961b-20fc-4805-9fd9-18e743036b20
# ╠═bc73915a-b65c-4a3a-83a9-4b4b1d9b6b3f
# ╟─59177c93-6ed8-4f1b-9047-7dacbd0d0fc2
# ╠═b21a3c67-8854-4f4a-bc7f-c611127151f7
# ╠═b6f57929-e80a-49df-a5da-3688b2bdfd97
# ╠═4d8b1cbc-7147-400d-b922-8cb99d15c4cf
# ╟─e0b880d3-c755-4cf2-9d83-efb05dc7de0d
# ╠═b996020b-b910-42d5-adf8-586678a28615
# ╟─f3f605ef-ecc1-43d7-8570-746b6196b34b
# ╠═dd982018-c9dc-48e3-88ca-0215c520e930
# ╠═f2b83b10-5b90-4ffa-8b1c-07ba797b2d3b
# ╠═b25d55fc-06ca-43a8-ad09-0d37dc4e922e
# ╟─abf14fd7-1d6f-4d23-90d2-c2c30f0754cd
# ╟─a8a79e4c-e3f0-4147-80ce-732bc8c185f6
# ╠═8b94d99c-d17f-442d-ac48-544ff11636b0
# ╠═d99a3632-59bb-47fd-a2fd-165494f37985
# ╟─309e43e4-26b3-4370-958f-e776e2364031
# ╠═a4ece2bd-9c80-4654-abe1-563da8f5d78b
# ╟─8b920d78-0dbe-4cff-83be-4c1abda9e92c
# ╠═7454e22a-8e92-4136-804b-02e8f507420c
# ╟─ee64f7a5-4d23-4a06-a9b6-11043b4b178f
# ╠═14aa4bf5-9f01-4e79-b3f0-e49daeae12da
# ╠═05be06e6-5bb7-4c31-acb7-bab97652b557
# ╟─14f0f686-1a41-4eb8-9354-e004505b34b2
# ╠═defe15c2-7153-46e9-9ece-4a9be738be3b
# ╟─976688f3-5748-4435-b842-7f402362babc
# ╟─83c56718-06b7-4953-b831-c02c069cf890
# ╠═ef1acd1d-4ddd-4532-97a1-b41bba8a7b75
# ╟─7f8243ce-dd87-4214-8122-cbab194e9c30
# ╠═7f6ed435-1972-419c-a27c-0bd69a47faa9
# ╠═ddf7bfe6-43bd-4acb-aab8-3c91aaa7cde6
# ╠═84086422-923c-4235-aedf-3f8de6683ee5
# ╟─35ac1d21-1026-432a-bd49-91efd71ec6c6
# ╟─920d4a27-0b5c-487e-9e43-32f3ec26c4a1
# ╠═9922b0a6-dad2-40ba-beb5-235d32d08086
# ╠═f97e3e4d-99c7-445c-8a58-18d15f5e3d2d
# ╟─a455455d-97a1-4583-85a3-4d2599ace0ef
# ╠═edc51a4b-0081-4237-bc0a-56abf7252efc
# ╠═bb8bc581-bd2b-4d45-95fd-9f7be09b0383
# ╠═8af35e77-1ea0-4f79-97d9-11749fdd8765
# ╟─44e0cc5f-a3d6-4b19-8621-ec1dcec4f8c8
# ╟─9a33e66c-0831-43de-bb10-c8324b62a95a
# ╠═c0e51846-4c1f-45db-a930-6a3c93b9c742
# ╟─50d8de79-d55a-4958-b310-28752f11665d
# ╠═159827aa-aa38-4f3d-b1f3-8e138d72cde4
# ╠═d666c6ce-9831-4902-81d0-29de1c6c0c2e
# ╟─79df769a-2b67-49d6-86a8-b1220e7dd993
# ╟─29a5884f-5e14-4371-8c73-26c67e248aad
# ╠═fb3175b4-63f1-4723-8381-fa4dd46245b0
# ╠═a77e4af8-1d18-4391-b509-7267d23e3bfe
# ╟─eb1d514a-bfae-4132-adf5-a4fba73f63e0
# ╟─f5761099-b753-48a1-ac93-d3c1ef5ee0f1
# ╠═6d068573-9f54-4c46-8c5b-08ca7f93a551
# ╟─0d0a3c5f-d891-4983-a120-1950f8ce45af
# ╠═7b534287-d060-4216-968f-181eeb68ded6
# ╠═16bd1844-f8b3-4811-9c96-18d481593bff
# ╠═f6c3d52c-f7a4-4e3a-a0e9-db0740187938
# ╟─116480c6-33d6-458e-b589-ea824ffe1ea7
# ╠═df72114f-6e2e-4af0-b80d-d443b8e7c560
# ╟─06a20ad0-24bc-4f6d-b7ec-d22df054c28f
# ╟─62997e2b-3177-4949-97d2-19f356829171
# ╟─5fd8e8c7-ad55-4dce-832f-0c49a74d691c
# ╠═439b946f-188d-48e9-aa30-3adacd6b52f5
# ╠═c6f9c07f-7a7a-4103-b699-623a6108906f
# ╠═6f0e4b4e-d8bc-48b1-8c86-4e4f103e05ca
# ╠═6b96702c-c3f4-41bb-8a4a-cd227db71a92
# ╠═8064b33b-c00b-4dd6-9fe0-a2d7bc14132a
# ╟─bb140735-a485-41ef-ac6b-f8fbfa429d9e
# ╠═e06912a1-14e3-4664-9967-4a6f624a3c32
# ╠═f1c32574-dd94-425c-890a-7692524ab5f7
# ╟─07a46c0f-bb89-446e-a6ad-f78efa078a1b
# ╠═982537c6-7189-403c-aca4-819e93604747
# ╠═f34199b8-e39e-44cd-85e7-0fb362574b21
# ╟─30905821-8eb2-476f-bd32-65f7ee9a6fcc
# ╠═7fec9962-bd09-4a17-8602-7a30c35f8aa3
# ╠═2f13d9e1-70c5-4b49-8e7a-82298640d2a3
# ╠═aeadfda5-b3e5-486e-a255-73b1174d10c0
# ╟─0bef159a-69fa-4af9-8fe5-8ee4977e02f0
# ╠═2e8c063e-a9cc-49c4-b428-ee32de46ddd9
# ╟─39cbd840-3821-4255-bccc-a837fcbe3c4f
# ╟─90956254-e8f6-4284-9691-652311dc4f2f
# ╠═c17da9ad-b9ed-46c5-a74e-1ad5466a27f9
# ╠═28a73360-59f0-4ff6-a5bc-b51dfd7e2c1a
# ╟─b7b53e73-66b7-494a-a376-e00244d1dbb8
# ╠═e11e00a1-2b3d-453a-8fe7-d2127fd67838
# ╠═670a27bb-0c36-4a19-9b9d-0c92e4f9df19
# ╠═6ca53978-fa5c-4ec9-93a0-eafd353fe793
# ╠═4548b4ae-1234-4e2e-9171-8d0c8eab64b8
# ╠═6adc6c7b-df5e-4147-92df-68a147ba7430
# ╟─3d263527-0937-4ecf-9545-611abcda1021
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
