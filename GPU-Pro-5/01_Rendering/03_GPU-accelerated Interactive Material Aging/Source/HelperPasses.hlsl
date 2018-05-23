#ifndef __HELPER_PASSES__
#define __HELPER_PASSES__

// This file is included in the atlas material

// some of the functions that are not directly involved in the material aging
// were removed from the effect to increate readablity

	pass DebugOutput
	{
		SetVertexShader( CompileShader( vs_4_0, VS_PassThrough() ) );
		SetHullShader( 0 );
		SetDomainShader( 0 );
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_4_0, PS_DebugOutput() ) );
	}

	pass Dilatation
	{
		SetVertexShader( CompileShader( vs_4_0, VS_PassThrough() ) );
		SetHullShader( 0 );
		SetDomainShader( 0 );
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_4_0, PS_Dilatation() ) );
	}

#endif