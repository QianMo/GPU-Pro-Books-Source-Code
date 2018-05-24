/*
Copyright (c) 2016 SIDIA

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

Shader "BRSGraphics/HalftoneBook"
{
	Properties
	{
		_DiffuseMap("DiffuseMap", 2D) = "white" {}
		_LightRamp("LightRamp", 2D) = "white" {}
		_RepeatRate("HT RepeatRate", Float) = 150
		_HalftoneScale("HT Scale", Float) = 1

		_HalftoneCutoff("HT Cutoff", Range(0.0, 1.0)) = 0.05
		_HalftonePatternAngle("HT Pattern Angle", Float) = 0.0
		_HalftoneColor("HT Color Shadow", Color) = (0.5,0.5,0.5,1)
		_MinFadeDistance("HT MinFadeDistanceFromCharacter", Float) = 0.0
		_MaxFadeDistance("HT MaxFadeDistanceFromCharacter", Float) = 1000.0

		_OutlineThicknessMap("Outline Thickness Map", 2D) = "white" {}
		_OutlineScale("OutlineScale", Range(0.0, 1.0)) = 1.0
		_OutlineMinOffset("OutlineThinness", Range(0.0, 1.0)) = 0.0
		_OutlineMaxOffset("OutlineThickness", Range(0.0, 1.0)) = 0.3
		_OutlineColor("OutlineColor", Color) = (1.0, 1.0, 1.0, 1.0)

		_EmissionMap("Emission Map", 2D) = "white" {}
		_EmissionIntensity("Emission Intensity", Float) = 1

		[MaterialToggle]_EnableSoftlight("Enable Softlight", Float) = 1.0
		[MaterialToggle]_EnableHalftone("Enable Halftone", Float) = 1.0
		[MaterialToggle]_EnableOutline("Enable Outline", Float) = 1.0
		[MaterialToggle]_EnableEmission("Enable Emission", Float) = 1.0
	}

	SubShader
	{
		Tags
		{
			//"Queue" = "Transparent" // Uncomment to make outline work with unity skybox
			"RenderType" = "Opaque"
		}
		LOD 400

		Pass
		{
			Name "Outline"

			Cull Front
			ZWrite Off
			Fog{ Mode Off }

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#include "UnityCG.cginc"
			#include "ShaderUtilities.cginc"
			#pragma fragmentoption ARB_precision_hint_fastest
			#pragma target 3.0

			sampler2D _OutlineThicknessMap;
			uniform half _OutlineScale;
			uniform half _OutlineMinOffset;
			uniform half _OutlineMaxOffset;
			uniform half4 _OutlineColor;
			
			half _EnableOutline;

			struct VertexInput
			{
				half4 vertex : POSITION;
				half3 normal : TANGENT;
				half2 texcoord : TEXCOORD0;
			};

			struct VertexOutput
			{
				half4 pos : SV_POSITION;
			};

			VertexOutput vert(VertexInput v)
			{
				VertexOutput o = (VertexOutput)0;
				half thicknessOffset = tex2Dlod(_OutlineThicknessMap, half4(v.texcoord.xy, 0.0, 0.0)).r;
				half outlineWidth = clamp(thicknessOffset * _OutlineScale, _OutlineMinOffset, _OutlineMaxOffset);

				outlineWidth *= step(0.5, _EnableOutline);
				o.pos = mul(UNITY_MATRIX_MVP, half4(v.vertex.xyz + (v.normal * outlineWidth), 1.0));

				return o;
			}

			fixed4 frag(VertexOutput i) : COLOR
			{
				return _OutlineColor;
			}
			ENDCG
		}

		Pass
		{
			Name "Halftone"

			Fog{ Mode Off }

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#include "UnityCG.cginc"
			#include "ShaderUtilities.cginc"
			#pragma fragmentoption ARB_precision_hint_fastest
			#pragma target 3.0

			uniform sampler2D _DiffuseMap; uniform half4 _DiffuseMap_ST;
			uniform half _RepeatRate;

			uniform sampler2D _LightRamp; uniform half4 _LightRamp_ST;
			uniform half4 _HalftoneColor;
			uniform half _HalftoneScale;
			uniform half _HalftoneCutoff;
			uniform half _MinFadeDistance;
			uniform half _MaxFadeDistance;
			uniform half _HalftonePatternAngle;

			uniform sampler2D _EmissionMap; uniform half4 _EmissionMap_ST;
			uniform half _EmissionIntensity;

			half _EnableSoftlight;
			half _EnableHalftone;
			half _EnableEmission;
			
			struct VertexInput
			{
				half4 vertex : POSITION;
				half3 normal : NORMAL;
				half2 texcoord0 : TEXCOORD0;
			};

			struct VertexOutput
			{
				half4 pos : SV_POSITION;
				half2 uv0 : TEXCOORD0;
				half4 posWorld : TEXCOORD1;
				half3 normalDir : TEXCOORD2;
				half4 screenPos : TEXCOORD3;
			};

			VertexOutput vert(VertexInput v)
			{
				VertexOutput o = (VertexOutput)0;
				o.uv0 = v.texcoord0;

				// Tangent Space Setup
				o.normalDir = mul(unity_ObjectToWorld, half4(v.normal,0)).xyz;

				// Needed to compute viewdir in frag shader.
				o.posWorld = mul(unity_ObjectToWorld, v.vertex);

				o.pos = mul(UNITY_MATRIX_MVP, v.vertex);
				o.screenPos = o.pos;
				return o; 
			}

			fixed4 frag(VertexOutput i) : COLOR
			{
				half2 screenUV = NormalizeScreenSpace(i.screenPos);
				screenUV = RotateUV(screenUV, _HalftonePatternAngle);

				half3 lightDirection = normalize(_WorldSpaceLightPos0.xyz);
				half3 normalDirection = normalize(i.normalDir);

				/////////////////////////////////////////////////////////////////////////////////////
				// Halftone Factor setups by using Dots, Ramps and Distance Scale					/
				/////////////////////////////////////////////////////////////////////////////////////
				half repeatRate = _RepeatRate;
				half halftoneScale = _HalftoneScale;

				half camDistance = length(i.posWorld - _WorldSpaceCameraPos);

				half halftoneAlphaFade = smoothstep(_MaxFadeDistance, _MinFadeDistance, camDistance);

				/////////////////////////////////////////////////////////////////////////////////////
				// Apply Lambert Light term using a lightRamp                                       /
				/////////////////////////////////////////////////////////////////////////////////////
				half3 diffuseMap = tex2dLinear(_DiffuseMap, TRANSFORM_TEX(i.uv0, _DiffuseMap)).rgb;

				// We use the NdotL to sample a Ramp in order to get the diffusecolor
				half NdotL = saturate(dot(normalDirection, lightDirection));
				half lambertian = tex2dLinear(_LightRamp, half2(NdotL, 0.0)).r * 0.5; // <- this 0.5 multiplacation can be done in the ramp texture

				/////////////////////////////////////////////////////////////////////////////////////
				// Color Composition                                                                /
				// Light Model = lerp(diffuse, halftoneShadow) + halftoneSpecular + halftoneLight   /
				// Where: halftoneShadow is greater when Ramp(dot(N.L)) is close to zero.           /
				//        halftoneLight is greater when Ramp(dot(N.L)) is close to one.             /
				//        halftoneSpecular is present in specular highlights                        /
				///////////////////////////////////////////////////////////////////////////////////// 

				half halftoneFactor = (1.0 - NdotL) * _HalftoneScale;

				half halftoneCutoff = step(_HalftoneCutoff, halftoneFactor);
				half halftoneDebug = step(0.5, _EnableHalftone);

				half3 diffuse = half3(lambertian, lambertian, lambertian);
				half halftone = HalfTone(repeatRate, halftoneFactor, screenUV) * halftoneCutoff * halftoneAlphaFade * halftoneDebug;
				
				half3 halftoneColor = lerp(diffuse, _HalftoneColor, halftone);

				/////////////////////////////////////////////////////////////////////////////////////
				// Softlight
				/////////////////////////////////////////////////////////////////////////////////////                
				half enableSoftlight = step(0.5, _EnableSoftlight);
				half3 softlight = half3(Softlight(halftoneColor.r, diffuseMap.r), Softlight(halftoneColor.g, diffuseMap.g), Softlight(halftoneColor.b, diffuseMap.b));
				half3 finalColor = lerp(diffuseMap, softlight, enableSoftlight);

				/////////////////////////////////////////////////////////////////////////////////////
				// Emissive
				/////////////////////////////////////////////////////////////////////////////////////
				half4 _EmissionMap_var = tex2dLinear(_EmissionMap,TRANSFORM_TEX(i.uv0, _EmissionMap));
				half3 emissive = (_EmissionMap_var.rgb * _EmissionIntensity);

				half emissionDebug = step(0.5, _EnableEmission);

				finalColor += diffuseMap * emissive * emissionDebug;			

				// DEBUGS
				// Lambert + Ramp
				//finalColor = diffuse;

				// Albedo (Softlight bottom layer)
				//finalColor = diffuseMap;

				// Albedo * Lambert
				//finalColor = diffuseColor * diffuseMap;

				// Halftone Color (Softlight top layer)
				//finalColor = halftoneColor;

				// Final Color
				//finalColor = softlight;

				return gammaConvert(fixed4(finalColor, 1));				
			}
			ENDCG
		}
	}
	FallBack "Diffuse"
}