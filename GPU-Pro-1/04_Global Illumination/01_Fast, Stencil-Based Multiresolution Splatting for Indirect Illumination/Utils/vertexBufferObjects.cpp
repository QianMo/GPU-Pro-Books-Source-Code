/***************************************************************************/
/* vertexBufferObjects.cpp                                                 */
/* -----------------------                                                 */
/*                                                                         */
/* A basic class for using vertex buffer objects.                          */
/*                                                                         */
/* Chris Wyman (04/25/2008)                                                */
/***************************************************************************/

#include "vertexBufferObjects.h"

#ifndef BUFFER_OFFSET
#define BUFFER_OFFSET(x)   ((GLubyte*) 0 + (x))
#endif

VertexBuffer::VertexBuffer() : 
  enabled(false), validBuffers(VBO_INVALID), idxID(0) 
{
	for (int i=0; i < VBO_MAX_BUFFERS; i++) 
		bufID[i] = 0;
}




void VertexBuffer::EnableVertexState( void )
{
	if (!bufID[0]) return;
	glEnableClientState( GL_VERTEX_ARRAY );
	glBindBuffer( GL_ARRAY_BUFFER, bufID[0] );
	glVertexPointer( bufSz[0], bufType[0], bufStride[0], BUFFER_OFFSET(0) );
}

void VertexBuffer::EnableNormalState( void )
{
	if (!bufID[1]) return;
	glEnableClientState( GL_NORMAL_ARRAY );
	glBindBuffer( GL_ARRAY_BUFFER, bufID[1] );
	glNormalPointer( bufType[1], bufStride[1], BUFFER_OFFSET(0) );
}

void VertexBuffer::EnableColorState( void )
{
	if (!bufID[2]) return;
	glEnableClientState( GL_COLOR_ARRAY );
	glBindBuffer( GL_ARRAY_BUFFER, bufID[2] );
	glColorPointer( bufSz[2], bufType[2], bufStride[2], BUFFER_OFFSET(0) );
}

void VertexBuffer::Enable2ndColorState( void )
{
	if (!bufID[3]) return;
	glEnableClientState( GL_SECONDARY_COLOR_ARRAY );
	glBindBuffer( GL_ARRAY_BUFFER, bufID[3] );
	glSecondaryColorPointer( bufSz[3], bufType[3], bufStride[3], BUFFER_OFFSET(0) );
}

void VertexBuffer::EnableFogState( void )
{
	if (!bufID[4]) return;
	glEnableClientState( GL_FOG_COORDINATE_ARRAY );
	glBindBuffer( GL_ARRAY_BUFFER, bufID[4] );
	glFogCoordPointer( bufType[4], bufStride[4], BUFFER_OFFSET(0) );
}

void VertexBuffer::EnableEdgeState( void )
{
	if (!bufID[5]) return;
	glEnableClientState( GL_EDGE_FLAG_ARRAY );
	glBindBuffer( GL_ARRAY_BUFFER, bufID[5] );
	glEdgeFlagPointer( bufStride[5], BUFFER_OFFSET(0) );
}

void VertexBuffer::EnableClientTextureState( GLuint texUnit )
{
	if (!bufID[6+texUnit]) return;
	glActiveTexture( texUnit );
	glEnableClientState( GL_TEXTURE_COORD_ARRAY );
	glBindBuffer( GL_ARRAY_BUFFER, bufID[6+texUnit] );
	glTexCoordPointer( bufSz[6+texUnit], bufType[6+texUnit], 
		               bufStride[6+texUnit], BUFFER_OFFSET(0) );
}

void VertexBuffer::EnableInterleavedState( void )
{
	if (!bufID[0]) return;
	glBindBuffer( GL_ARRAY_BUFFER, bufID[0] );
	glInterleavedArrays( bufType[0], bufStride[0], BUFFER_OFFSET(0) );
}

void VertexBuffer::DisableClientTextureState( GLuint texUnit )
{
	glActiveTexture( texUnit );
	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
}


void VertexBuffer::Enable( void )
{
	if (enabled) return;

	if ( validBuffers & VBO_INTERLEAVED_DATA ) 
		EnableInterleavedState();
	else
	{
		if ( validBuffers & VBO_VERTEX_DATA )      EnableVertexState();
		if ( validBuffers & VBO_NORMAL_DATA )      EnableNormalState();
		if ( validBuffers & VBO_COLOR_DATA )       EnableColorState();
		if ( validBuffers & VBO_2NDCOLOR_DATA )    Enable2ndColorState();
		if ( validBuffers & VBO_FOG_DATA )         EnableFogState();
		if ( validBuffers & VBO_EDGE_FLAG_DATA )   EnableEdgeState();
		if ( validBuffers & VBO_TEXCOORD0_DATA )   EnableClientTextureState( GL_TEXTURE0 );
		if ( validBuffers & VBO_TEXCOORD1_DATA )   EnableClientTextureState( GL_TEXTURE1 );
		if ( validBuffers & VBO_TEXCOORD2_DATA )   EnableClientTextureState( GL_TEXTURE2 );
		if ( validBuffers & VBO_TEXCOORD3_DATA )   EnableClientTextureState( GL_TEXTURE3 );
		if ( validBuffers & VBO_TEXCOORD4_DATA )   EnableClientTextureState( GL_TEXTURE4 );
		if ( validBuffers & VBO_TEXCOORD5_DATA )   EnableClientTextureState( GL_TEXTURE5 );
		if ( validBuffers & VBO_TEXCOORD6_DATA )   EnableClientTextureState( GL_TEXTURE6 );
		if ( validBuffers & VBO_TEXCOORD7_DATA )   EnableClientTextureState( GL_TEXTURE7 );
	}

	// If this is indexed data, setup the ELEMENT_ARRAY_BUFFER
	if (idxID > 0)
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, idxID );

	enabled = true;
}

void VertexBuffer::Disable( void )
{
	if (!enabled) return;

	if (idxID > 0)
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

	// Always disable these, since interleaved textures will enable some, and
	//   we won't know which unless the class incorporates knowledge of the
	//   different interleaved constants (e.g., GL_T4F_V4F )
	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableClientState( GL_NORMAL_ARRAY );
	glDisableClientState( GL_COLOR_ARRAY );
	glDisableClientState( GL_SECONDARY_COLOR_ARRAY );
	glDisableClientState( GL_FOG_COORDINATE_ARRAY );
	glDisableClientState( GL_EDGE_FLAG_ARRAY );

	// Seems like a duplicate, but it allows disabling textures using interleaved
	//   arrays without the side effect of changing the active texture and searching
	//   blindly through all 7 textures to simply disable one.
	glDisableClientState( GL_TEXTURE_COORD_ARRAY ); 
	if ( validBuffers & VBO_TEXCOORD0_DATA )  DisableClientTextureState( GL_TEXTURE0 );
	if ( validBuffers & VBO_TEXCOORD1_DATA )  DisableClientTextureState( GL_TEXTURE1 );
	if ( validBuffers & VBO_TEXCOORD2_DATA )  DisableClientTextureState( GL_TEXTURE2 );
	if ( validBuffers & VBO_TEXCOORD3_DATA )  DisableClientTextureState( GL_TEXTURE3 );
	if ( validBuffers & VBO_TEXCOORD4_DATA )  DisableClientTextureState( GL_TEXTURE4 );
	if ( validBuffers & VBO_TEXCOORD5_DATA )  DisableClientTextureState( GL_TEXTURE5 );
	if ( validBuffers & VBO_TEXCOORD6_DATA )  DisableClientTextureState( GL_TEXTURE6 );
	if ( validBuffers & VBO_TEXCOORD7_DATA )  DisableClientTextureState( GL_TEXTURE7 );

	// Set the current array buffer to 0.
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	enabled = false;
}


void VertexBuffer::EnableOnly( GLuint clientState )
{
	if (enabled) return;

	// We can't really selectively enable in interleaved mode....
	if ( validBuffers & VBO_INTERLEAVED_DATA ) 
		EnableInterleavedState();
	else
	{
		GLuint validBits = clientState & validBuffers;
		if ( validBits & VBO_VERTEX_DATA )      EnableVertexState();
		if ( validBits & VBO_NORMAL_DATA )      EnableNormalState();
		if ( validBits & VBO_COLOR_DATA )       EnableColorState();
		if ( validBits & VBO_2NDCOLOR_DATA )    Enable2ndColorState();
		if ( validBits & VBO_FOG_DATA )         EnableFogState();
		if ( validBits & VBO_EDGE_FLAG_DATA )   EnableEdgeState();
		if ( validBits & VBO_TEXCOORD0_DATA )   EnableClientTextureState( GL_TEXTURE0 );
		if ( validBits & VBO_TEXCOORD1_DATA )   EnableClientTextureState( GL_TEXTURE1 );
		if ( validBits & VBO_TEXCOORD2_DATA )   EnableClientTextureState( GL_TEXTURE2 );
		if ( validBits & VBO_TEXCOORD3_DATA )   EnableClientTextureState( GL_TEXTURE3 );
		if ( validBits & VBO_TEXCOORD4_DATA )   EnableClientTextureState( GL_TEXTURE4 );
		if ( validBits & VBO_TEXCOORD5_DATA )   EnableClientTextureState( GL_TEXTURE5 );
		if ( validBits & VBO_TEXCOORD6_DATA )   EnableClientTextureState( GL_TEXTURE6 );
		if ( validBits & VBO_TEXCOORD7_DATA )   EnableClientTextureState( GL_TEXTURE7 );
	}

	// If this is indexed data, setup the ELEMENT_ARRAY_BUFFER
	if (idxID > 0)
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, idxID );

	enabled = true;
}



