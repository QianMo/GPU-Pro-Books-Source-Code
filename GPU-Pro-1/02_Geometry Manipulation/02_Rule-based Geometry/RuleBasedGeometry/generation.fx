/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Milan Magdics
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

#ifndef __GENERATION_H
#define __GENERATION_H

#include "rulemanagement.fx"   // NOTE: this is auto-generated from the L-system description
#include "culling.fx"

//********************************************************
// GENERATION PHASE
//
// generates the next level of the procedural scene graph
//********************************************************

// VS of the geometry generation phase
Module vsGeneration( Module module )
{
	return module;
};

//**********************************************************
// GS of the geometry generation phase
//	- selects a rule
//	- sets the parameters of the modules in the successor
//	- appends the output stream with the new modules
//**********************************************************
[maxvertexcount(MAX_SUCCESSOR_LENGTH)]
void gsGeneration(point Module input[1], inout PointStream<Module> stream)
{
	// module culling
	if ( cullModule( input[0] ))
	{
		return;
	}

	// checking termination
	if (input[0].terminated)
	{
		stream.Append(input[0]);
		return;
	}

	// Selecting a rule and getting it's successor length (number of modules in it)
	int successor_length, rule_id;
	selectRule( input[0], successor_length, rule_id );
	
	// Getting the successor modules and streaming them out
	for ( int i = __GR_SYMBOLID_START; i < successor_length + __GR_SYMBOLID_START; ++i )
	{
		stream.Append(getNextModule(input[0], rule_id, i));
	}
}

DepthStencilState DisableDepthTestWrite
{
    DepthEnable = FALSE;
    DepthWriteMask = ZERO;
};

technique10 generateGeometry
{
    pass P0
    {
        SetVertexShader ( CompileShader( vs_4_0, vsGeneration() ) );
        SetGeometryShader( 
	        ConstructGSWithSO( CompileShader( gs_4_0, gsGeneration()), __GR_GENERATION_SO_ARG ) );
        SetPixelShader( NULL );

		SetDepthStencilState( DisableDepthTestWrite, 0);
    }
}

#endif