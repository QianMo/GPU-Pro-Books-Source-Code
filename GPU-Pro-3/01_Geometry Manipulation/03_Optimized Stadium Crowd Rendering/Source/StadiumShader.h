/**
 *	@file
 *	@brief		A simple vertex color shader for the stadium and pitch.
 *	@author		Alan Chambers
 *	@date		2011
**/

#ifndef STADIUMSHADER_H
#define STADIUMSHADER_H

const char stadium_shader[] =
"\
	struct VertexInput\n\
	{\n\
		float4 pos: 		POSITION;\n\
		float4 color:		COLOR;\n\
	};\n\
\n\
	struct FragmentInput\n\
	{\n\
		float4 pos : 		POSITION;\n\
		float4 color:		COLOR;\n\
	};\n\
\n\
	struct FragmentOutput\n\
	{\n\
		float4 color:		COLOR0;\n\
	};\n\
\n\
	float4 light_dir;\n\
	float4 light_col;\n\
	float4 light_amb;\n\
	float4x4 world;\n\
	float4x4 world_view_proj;\n\
\n\
	FragmentInput main_vs( VertexInput vs_in )\n\
	{\n\
		FragmentInput vs_out;\n\
\n\
// 		float3 normal = normalize( mul( float3x3( transpose( world ) ), n ) );\n\
// 		float n_dot_l = saturate( dot( normal, light_dir.xyz ) );\n\
// 		float3 diffuse = n_dot_l * light_col.xyz /* * shadow.x */;\n\
		vs_out.pos = mul( world_view_proj, vs_in.pos );\n\
		vs_out.color = vs_in.color;\n\
\n\
		return vs_out;\n\
	}\n\
\n\
	FragmentOutput main_fs( FragmentInput fs_in )\n\
	{\n\
		FragmentOutput fs_out;\n\
\n\
		fs_out.color = fs_in.color;\n\
\n\
		return fs_out;\n\
	}\n\
";

#endif
