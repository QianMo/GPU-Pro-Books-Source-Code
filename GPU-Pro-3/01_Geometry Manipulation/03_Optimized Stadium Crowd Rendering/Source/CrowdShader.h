/**
 *	@file
 *	@brief		A shader for the crowd that extrapolates each corner of a quadrilateral and maps
 *				a desired model. Per-pixel dynamic lighting through an ambient and directional
 *				light is also supported.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef CROWDSHADER_H
#define CROWDSHADER_H

const char crowd_shader[] =
"\
	sampler2D base_map :	TEXUNIT0;\n\
	sampler2D normal_map :	TEXUNIT1;\n\
\n\
	uniform float4 positions[16] : BUFFER[0];\n\
	uniform float4 properties[16] : BUFFER[1];\n\
\n\
	struct VertexInput\n\
	{\n\
		float4 pos: 		POSITION;\n\
		int id:				INSTANCEID;\n\
	};\n\
\n\
	struct FragmentInput\n\
	{\n\
		float4 pos : 		POSITION;\n\
		float2 uv :			TEXCOORD0;\n\
	};\n\
\n\
	struct FragmentOutput\n\
	{\n\
		float4 color :		COLOR0;\n\
	};\n\
\n\
	float density;\n\
	float mirror;\n\
	float4 light_dir;\n\
	float4 light_col;\n\
	float4 light_amb;\n\
	float4x4 world;\n\
	float4x4 view;\n\
	float4x4 proj;\n\
\n\
	FragmentInput main_vs( VertexInput vs_in )\n\
	{\n\
		FragmentInput vs_out;\n\
\n\
		float4 pos = positions[vs_in.id];\n\
		float4 prop = properties[vs_in.id];\n\
		pos = mul( world, pos );\n\
		pos = mul( view, pos ) + vs_in.pos * float4( 1.12, 2.32, 0.0, 0.0 );\n\
		pos = mul( proj, pos );\n\
		float2 uv = abs( vs_in.pos.xy - float2( mirror, 0.0 ) );\n\
		float2 seat_uv = uv * float2( 0.1, 0.25 );\n\
		float2 dude_uv = float2( prop.z, 0.0 ) + seat_uv;\n\
		uv = lerp( dude_uv, seat_uv, step( density, prop.x ) );\n\
		vs_out.pos = pos;\n\
		vs_out.uv = uv;\n\
\n\
		return vs_out;\n\
	}\n\
\n\
	FragmentOutput main_fs( FragmentInput fs_in )\n\
	{\n\
		FragmentOutput fs_out;\n\
\n\
   		float3 n = tex2D( normal_map, fs_in.uv ).xyz * 2 - 1;\n\
		float3 normal = normalize( mul( float3x3( transpose( world ) ), n ) );\n\
  		float n_dot_l = saturate( dot( normal, light_dir.xyz ) );\n\
  		float3 diffuse = n_dot_l * light_col.xyz /* * shadow.x */;\n\
		fs_out.color = tex2D( base_map, fs_in.uv );\n\
  		fs_out.color.rgb *= light_amb.xyz + diffuse;\n\
\n\
		return fs_out;\n\
	}\n\
";

#endif
