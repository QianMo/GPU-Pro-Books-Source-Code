/***************************************************************************/
/* vertexBufferObjects.h                                                   */
/* -----------------------                                                 */
/*                                                                         */
/* A basic class for using vertex buffer objects.                          */
/*                                                                         */
/* Chris Wyman (04/25/2008)                                                */
/***************************************************************************/

#ifndef ___VERTEXBUFFER_OBJECT_H
#define ___VERTEXBUFFER_OBJECT_H

#include "Utils/GLee.h"

// Constants to specify what sort of data (client state) is stored in
//   the vertex buffer object.
#define VBO_INVALID				    0x00000000
#define VBO_INTERLEAVED_DATA        0x00000001
#define VBO_VERTEX_DATA	            0x00000002
#define VBO_NORMAL_DATA             0x00000004
#define VBO_COLOR_DATA              0x00000008
#define VBO_2NDCOLOR_DATA           0x00000010
#define VBO_FOG_DATA                0x00000020
#define VBO_TEXCOORD0_DATA          0x00000040
#define VBO_TEXCOORD1_DATA          0x00000080
#define VBO_TEXCOORD2_DATA          0x00000100
#define VBO_TEXCOORD3_DATA          0x00000200
#define VBO_TEXCOORD4_DATA          0x00000400
#define VBO_TEXCOORD5_DATA          0x00000800
#define VBO_TEXCOORD6_DATA          0x00001000
#define VBO_TEXCOORD7_DATA          0x00002000
#define VBO_EDGE_FLAG_DATA          0x00004000


// Number depends on the number of VBO_* constants below (though VBO_INVALID
//    and VBO_INTERLEAVED_DATA are not included in the count)
#define VBO_MAX_BUFFERS             14




class VertexBuffer
{
private:
	GLuint  bufID    [ VBO_MAX_BUFFERS ];
	GLenum  bufType  [ VBO_MAX_BUFFERS ];
	GLint   bufSz    [ VBO_MAX_BUFFERS ];
	GLsizei bufStride[ VBO_MAX_BUFFERS ];

	GLuint idxID, idxType, idxSz, idxStride;

	GLuint validBuffers;
	bool enabled;

	void EnableVertexState( void );
	void EnableNormalState( void );
	void EnableColorState( void );
	void Enable2ndColorState( void );
	void EnableFogState( void );
	void EnableEdgeState( void );
	void EnableInterleavedState( void );
	void EnableClientTextureState( GLuint texUnit );
	void DisableClientTextureState( GLuint texUnit );

public:
	// Setup the VBO
	VertexBuffer();
	~VertexBuffer();

	// Enable or disable the VBO for drawing.  When using Enable() & Disable()
	//  they must come in pairs, and they enable/diable all the client state
	//  used by the underlying vertex array.  
	// Note: These may change the state of glActiveTexture()
	void Enable( void );  
	void Disable( void );

	// Enable or disable the VBO for drawing.  When using EnableOnly() and 
	//  DisableOnly() they must come in pairs, and they enable/diable all 
	//  client state specified as a paramter that also is valid in the
	//  underlying vertex array.  OR various VBO_* constants together as input
	// Note: These may change the state of glActiveTexture()
	void EnableOnly( GLuint clientState=VBO_VERTEX_DATA );
	inline void DisableOnly( void ) { Disable(); }
	
	// Draw the geometry by calling the correct glDraw* command.  Please note
	//  these functions can be called *either* between an Enable/Disable
	//  pair or outside of them (they check the class enabled flag before
	//  drawing, enabling client state if necessary).  The enable state after
	//  the call is left the same as it was prior to the Draw() command.
	void Draw( void );
	void DrawOnly( GLuint clientState=VBO_VERTEX_DATA );

};




#endif

