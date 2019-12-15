Shader "Custom/ShadowReceiverPBR" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		_MainTex ("Albedo (RGB)", 2D) = "white" {}
		_SpecularTex ("Specular", 2D) = "white" {}
		_NormalTex ("Normal", 2D) = "bump" {}
		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0
		_EclipseOccluder ("Eclipse occluder", Range(0.0, 1.0)) = 0.0
	}
	SubShader {
		Tags { "RenderType"="Opaque" }

		CGPROGRAM
		#pragma surface surf StandardDefaultGI fullforwardshadows
		#pragma target 5.0
		#include "UnityPBSLighting.cginc"

		uniform half _Glossiness;
		uniform half _Metallic;
		uniform fixed4 _Color;

		uniform sampler2D _MainTex;
		uniform sampler2D _SpecularTex;
		uniform sampler2D _NormalTex;
		
		uniform sampler2D _BokehShape;
		uniform sampler2D _ShadowBokeh;
		uniform sampler2D _ShadowDepth;

		uniform float _BokehSize;
		uniform float _BokehMaxDistance;
		uniform float _BokehIntensity;

		struct Input 
		{
			float2 uv_MainTex;
			float2 uv_SpecularTex;
			float3 worldPos;
			float3 worldNormal;
			float3 viewDir;
			INTERNAL_DATA
		};
		
		struct PinholeData
		{
			float2 position; // The uv position of the pinhole in the shadowmap
			float meanDepth; // The depth of the pinhole in the shadowmap
			float rawDepth; // The estimated depth of the neighborhood
			float intensity; // The pinhole intensity
			uint next; // TODO: Remove
		}; 

		uniform uint PinholeGridSize;
		uniform uint PinholesPerCell;

		uniform float4x4 _LightTransform;
		uniform float4x4 _BokehRotation;
		uniform float3 _LightPosition;

		uniform float _EclipseOccluder;

		float2 WorldPositionToShadow(float3 pos)
		{
			float4 lightPos = mul(_LightTransform, float4(pos, 1.0));
			return (lightPos.xy / lightPos.w) * .5 + .5;
		}
	
#ifdef SHADER_API_D3D11
		uniform StructuredBuffer<PinholeData> g_TemporalPinholesBuffer : register(t1);
		uniform StructuredBuffer<uint> g_TemporalPinholesCountBuffer : register(t2);

		uint GetPinholeCount(float2 uv)
		{
			uint2 p = uint2(uv.x * PinholeGridSize, uv.y * PinholeGridSize);
			uint bufferIndex = p.y * PinholeGridSize + p.x;
			return g_TemporalPinholesCountBuffer.Load(bufferIndex);
		}

		// #define PROCEDURAL
		#define DEPTH_PROXIMITY .75

		float CollectBokehFromGridCellCompact(uint2 p, float2 uv, float currentDepth)
		{
			uint bufferIndex = p.y * PinholeGridSize + p.x;
			int bufferOffset = bufferIndex * PinholesPerCell;
			int count = g_TemporalPinholesCountBuffer.Load(bufferIndex);
			float accum = 0.0;
			
			[loop]
			for(int i = 0; i < count; ++i)
			{
				PinholeData pinhole = g_TemporalPinholesBuffer[bufferOffset + i];
				float proximity = currentDepth - pinhole.rawDepth;

				if(proximity < DEPTH_PROXIMITY)
				{
					// d_r = currentDepth
					// d_p = pinhole.meanDepth
					float CoC = saturate(abs(currentDepth - pinhole.meanDepth) * _BokehSize);

					// Far pinholes lose contribution, regardless of their shape
					float d = distance(pinhole.position, uv);
					float intensity = pinhole.intensity * (1.0 - saturate(d / _BokehMaxDistance));

				#ifdef PROCEDURAL
					float2 occluderUV = uv + float2(_EclipseOccluder * 2.0 - 1.0, pow(_EclipseOccluder * 2.0 - 1.0, 4.0) * 4.5) * .02;
					accum += smoothstep(CoC, -CoC * .2, d) * smoothstep(CoC * .35, CoC * .9, distance(pinhole.position, occluderUV)) * intensity;
				#else
					float2 bokehUV = clamp((uv - pinhole.position) / (CoC * .5), -1.0, 1.0);
					bokehUV = mul(_BokehRotation, float4(bokehUV, 0.0, 0.0)).xy * 1.25;
					accum += tex2Dlod(_BokehShape, float4(bokehUV * .5 + .5, 0.0, 0.0)).x * intensity;
				#endif
				}
			}

			return accum;
		}

		// How many cells are iterated
		int CollectCellCount(float2 uv)
		{
			// We find the AABB for the maximum possible radius
			uint2 startCell = uint2((uv.x - _BokehMaxDistance) * PinholeGridSize, (uv.y - _BokehMaxDistance) * PinholeGridSize);
			uint2 endCell = uint2((uv.x + _BokehMaxDistance) * PinholeGridSize, (uv.y + _BokehMaxDistance) * PinholeGridSize);

			startCell = clamp(startCell, uint2(0,0), uint2(PinholeGridSize - 1, PinholeGridSize - 1));
			endCell = clamp(endCell, uint2(0,0), uint2(PinholeGridSize - 1, PinholeGridSize - 1));

			return (endCell.x + 1 - startCell.x) * (endCell.y + 1 - startCell.y);
		}

		// How many pinholes are actually compared
		int CollectTotalPinholeCount(float2 uv)
		{
			uint2 startCell = uint2((uv.x - _BokehMaxDistance) * PinholeGridSize, (uv.y - _BokehMaxDistance) * PinholeGridSize);
			uint2 endCell = uint2((uv.x + _BokehMaxDistance) * PinholeGridSize, (uv.y + _BokehMaxDistance) * PinholeGridSize);

			startCell = clamp(startCell, uint2(0,0), uint2(PinholeGridSize - 1, PinholeGridSize - 1));
			endCell = clamp(endCell, uint2(0,0), uint2(PinholeGridSize - 1, PinholeGridSize - 1));

			int count = 0;
			
			[loop]
			for(uint y = startCell.y; y <= endCell.y; y++)
			{
				[loop]
				for(uint x = startCell.x; x <= endCell.x; x++)
				{
					count += GetPinholeCount(float2(x / float(PinholeGridSize), y / float(PinholeGridSize)));
				}
			}

			return count;
		}

		float CollectBokehShadows(float3 pos)
		{
			float2 uv = WorldPositionToShadow(pos);

			// We find the AABB for the maximum possible radius
			uint2 startCell = uint2((uv.x - _BokehMaxDistance) * PinholeGridSize, (uv.y - _BokehMaxDistance) * PinholeGridSize);
			uint2 endCell = uint2((uv.x + _BokehMaxDistance) * PinholeGridSize, (uv.y + _BokehMaxDistance) * PinholeGridSize);

			startCell = clamp(startCell, uint2(0,0), uint2(PinholeGridSize - 1, PinholeGridSize - 1));
			endCell = clamp(endCell, uint2(0,0), uint2(PinholeGridSize - 1, PinholeGridSize - 1));

			float2 bokeh = 0.0;
			float currentDepth = distance(pos, _LightPosition);

			[loop]
			for(uint y = startCell.y; y <= endCell.y; y++)
			{
				[loop]
				for(uint x = startCell.x; x <= endCell.x; x++)
				{
					bokeh += CollectBokehFromGridCellCompact(uint2(x,y), uv, currentDepth);
				}
			}

			return saturate(bokeh * _BokehIntensity);
		}

		float ShowGrid(float2 uv)
		{
			int x = smoothstep(0.0, .01, fmod(uv.x * PinholeGridSize, 1.0));
			int y = smoothstep(0.0, .01, fmod(uv.y * PinholeGridSize, 1.0));
			return (1.0 - (x * y)) * step(abs(uv.x), 1.0) * step(abs(uv.y), 1.0);
		}

#endif

		float EvaluateBokehShadows(float3 worldPos)
		{
			float2 uv = WorldPositionToShadow(worldPos);
			float mask = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);

		#ifdef SHADER_API_D3D11
			return (mask > 0.0 ? CollectBokehShadows(worldPos) : 0.0);
		#else
			return 1.0;
		#endif
		}

		inline float4 LightingStandardDefaultGI(SurfaceOutputStandard s, half3 viewDir, UnityGI gi)
		{
			return LightingStandard(s, viewDir, gi);
		}

		// PCSS source from nvidia's implementation
		#ifdef SHADER_API_D3D11
		#define SHADOW_PCSS

		#define BLOCKER_SEARCH_NUM_SAMPLES 32
		#define PCF_NUM_SAMPLES 256
		#define NEAR_PLANE .5
		#define LIGHT_WORLD_SIZE 6.5
		#define LIGHT_FRUSTUM_WIDTH 30.0

		// Assuming that LIGHT_FRUSTUM_WIDTH == LIGHT_FRUSTUM_HEIGHT
		#define LIGHT_SIZE_UV (LIGHT_WORLD_SIZE / LIGHT_FRUSTUM_WIDTH)

		float PenumbraSize(float zReceiver, float zBlocker) //Parallel plane estimation
		{
			return (zReceiver - zBlocker) / zBlocker;
		}
		
		float2 R2seq(int n)
		{
			return frac(float2(n, n) * float2(0.754877666246692760049508896358532874940835564978799543103, 0.569840290998053265911399958119574964216147658520394151385));
		}

		void FindBlocker(out float avgBlockerDepth, out float numBlockers, float2 uv, float zReceiver )
		{
			//This uses similar triangles to compute what
			//area of the shadow map we should search
			float searchWidth = LIGHT_SIZE_UV * (zReceiver - NEAR_PLANE) / zReceiver;
			float blockerSum = 0;
			numBlockers = 0;

			for( int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; ++i )
			{
				float2 offset = R2seq(i) * 2.0 - 1.0;
				float shadowMapDepth = tex2Dlod(_ShadowDepth, float4(uv + offset * searchWidth, 0.0, 0.0)).r;
				
				if(shadowMapDepth < zReceiver)
				{
					blockerSum += shadowMapDepth;
					numBlockers++;
				}
			}

			avgBlockerDepth = blockerSum / numBlockers;
		}

		float PCF_Filter( float2 uv, float zReceiver, float filterRadiusUV )
		{
			float sum = 0.0f;

			for(int i = 0; i < PCF_NUM_SAMPLES; ++i)
			{
				float2 offset = R2seq(i) * 2.0 - 1.0;

				if(length(offset) < 1.0)
				{
					float d = tex2Dlod(_ShadowDepth, float4(uv + offset * filterRadiusUV, 0.0, 0.0)).r;
					sum += step(zReceiver, d);
				}
			}

			return sum / PCF_NUM_SAMPLES;
		}

		float PCSS (float2 uv, float zReceiver)
		{
			// STEP 1: blocker search
			float avgBlockerDepth = 0;
			float numBlockers = 0;

			FindBlocker(avgBlockerDepth, numBlockers, uv, zReceiver);

			//There are no occluders so early out (this saves filtering)
			if(numBlockers < 1)
				return 1.0f;
			
			// STEP 2: penumbra size
			float penumbraRatio = PenumbraSize(zReceiver, avgBlockerDepth);
			float filterRadiusUV = penumbraRatio * LIGHT_SIZE_UV * NEAR_PLANE / zReceiver;
			
			// STEP 3: filtering
			return PCF_Filter(uv, zReceiver, filterRadiusUV);
		}

		#endif

		#define SHADOW_PINHOLES

		inline void LightingStandardDefaultGI_GI(SurfaceOutputStandard s, UnityGIInput data, inout UnityGI gi)
		{
			#ifdef SHADOW_PINHOLES
				data.atten = saturate(data.atten + EvaluateBokehShadows(data.worldPos));
			#else
			#ifdef SHADOW_PCSS
				float2 uv = WorldPositionToShadow(data.worldPos);
				float currentDepth = distance(data.worldPos, _LightPosition);
				data.atten = PCSS(uv, currentDepth);
			#endif
			#endif

			LightingStandard_GI(s, data, gi);
		}

		float4 triplanar(sampler2D tex, float3 P, float3 N)
		{   
			float3 Nb = abs(N);
			
			float b = (Nb.x + Nb.y + Nb.z);
			Nb /= b;
			
			float4 c0 = tex2D(tex, P.xy) * Nb.z;
			float4 c1 = tex2D(tex, P.yz) * Nb.x;
			float4 c2 = tex2D(tex, P.xz) * Nb.y;
			
			return c0 + c1 + c2;
		}

		void surf (Input IN, inout SurfaceOutputStandard o) 
		{
			float scaleFactor = .2;
			float2 uv =  IN.worldPos.xy * .2;
			float4 normalTex = triplanar(_NormalTex, IN.worldPos * scaleFactor, WorldNormalVector(IN, float3(0.0, 0.0, 1.0)));
			o.Normal = UnpackNormal(normalTex);

			float3 worldNormal = WorldNormalVector (IN, o.Normal);

			o.Albedo = triplanar(_MainTex, IN.worldPos * scaleFactor, worldNormal) * _Color;
			o.Metallic = _Metallic;
			o.Smoothness = saturate(length(triplanar(_SpecularTex, IN.worldPos * scaleFactor, worldNormal)) * .5 * _Glossiness);
			o.Alpha = 1.0;
		}
		ENDCG
	}

	FallBack "Standard"
}
