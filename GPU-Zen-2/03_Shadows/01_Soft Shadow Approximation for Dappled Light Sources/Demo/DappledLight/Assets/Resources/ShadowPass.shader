Shader "Unlit/ShadowPass"
{	
	SubShader
	{
		Tags { "RenderType"="Opaque" "Queue"="Geometry" }
		Cull Off

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma target 5.0
			#pragma instancing_options assumeuniformscaling maxcount:50
			#pragma multi_compile_vertex LOD_FADE_PERCENTAGE LOD_FADE_CROSSFADE
			#pragma multi_compile_fragment __ LOD_FADE_CROSSFADE
			#pragma multi_compile_instancing
			#pragma shader_feature GEOM_TYPE_BRANCH GEOM_TYPE_BRANCH_DETAIL GEOM_TYPE_FROND GEOM_TYPE_LEAF GEOM_TYPE_MESH
			#pragma multi_compile_shadowcaster

			#define ENABLE_WIND
			#define GEOM_TYPE_LEAF

			#include "SpeedTreeCommon.cginc"
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float3 normal : NORMAL;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				V2F_SHADOW_CASTER;
				#ifdef SPEEDTREE_ALPHATEST
					float2 uv : TEXCOORD1;
				#endif
				UNITY_VERTEX_INPUT_INSTANCE_ID
				UNITY_VERTEX_OUTPUT_STEREO
				float3 wsPos : TEXCOORD5;
			};

			// uniform sampler2D  _MainTex;
			uniform float3 _LightPosition;
			
			v2f vert (SpeedTreeVB v)
			
			{
				v2f o;
				UNITY_SETUP_INSTANCE_ID(v);
				UNITY_TRANSFER_INSTANCE_ID(v, o);
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
				#ifdef SPEEDTREE_ALPHATEST
					o.uv = v.texcoord.xy;
				#endif
				OffsetSpeedTreeVertex(v, unity_LODFade.x);
				o.wsPos = mul(unity_ObjectToWorld, v.vertex).xyz;
				TRANSFER_SHADOW_CASTER_NORMALOFFSET(o)

				return o;
			}
			
			float4 frag (v2f i) : SV_Target
			{
				UNITY_SETUP_INSTANCE_ID(i);
				#ifdef SPEEDTREE_ALPHATEST
					clip(tex2D(_MainTex, i.uv).a * _Color.a - _Cutoff);
				#endif
				UNITY_APPLY_DITHER_CROSSFADE(i.pos.xy);
				
				float d = distance(i.wsPos, _LightPosition);
				return float4(d, d * d, 0.0, 0.0);
			}
			ENDCG
		}
	}

	
}
