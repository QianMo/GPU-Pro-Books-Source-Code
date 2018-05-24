/*

Copyright 2013,2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901

Sergio Ruiz and Benjamin Hernandez. 2015. A Parallel Solver for Markov Decision Process
in Crowd Simulations. Fourteenth Mexican International Conference on Artificial
Intelligence (MICAI), Cuernavaca, 2015, pp. 107-116.
DOI: 10.1109/MICAI.2015.23

*/

#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string>

#define DEMO_SHADER

//Enable one of these:
#undef DEMO_ALONE
#define DEMO_TEASER
#undef DEMO_LEMMINGS

#define CROWD_OK								0
#define CROWD_ERROR								1
#define CROWD_FBO_ERROR							2
#define CROWD_SHADER_ERROR						3

#define CUDA_PATHS

//Enable one of these:
#undef	MDPS_SQUARE_NOLCA
#undef	MDPS_HEXAGON_NOLCA
#define MDPS_SQUARE_LCA
#undef	MDPS_HEXAGON_LCA

#if defined MDPS_SQUARE_LCA || defined MDPS_SQUARE_NOLCA
	#define	DIRECTIONS	8
#elif defined MDPS_HEXAGON_LCA || defined MDPS_HEXAGON_NOLCA
	#define	DIRECTIONS	6
#else 
	#define DIRECTIONS	0
#endif

#define MDP_CHANNELS							1

//LCA macros:
#define LCA_RATIO								6
//#define PRINT_LCA_DATA

//Scenario macros:
//#define DRAW_SKYBOX
//#define DRAW_SCENARIO
#ifndef __SCENARIO_TYPES
#define __SCENARIO_TYPES
	enum SCENARIO_TYPE { ST_NONE, ST_O1P2, ST_EIFFEL, ST_TOWN, ST_MAZE, ST_CCM };
#endif

#ifndef __CROWD_POSITIONS
#define __CROWD_POSITIONS
	enum CROWD_POSITION{ CPOS_NONE, CPOS_TOP, CPOS_BOTTOM, CPOS_LEFT, CPOS_RIGHT, CPOS_CENTER };
#endif

#define DRAW_OBSTACLES
#ifndef __OBSTACLE_TYPES
#define __OBSTACLE_TYPES
	enum OBSTACLE_TYPE { OT_NONE, OT_BOX, OT_STATUE, OT_BUDDHA, OT_BARREL, OT_GLASSBOX };
#endif

//For CUDA debugging:
#undef CASIM_CUDA_PATH_DEBUG

//SLoD:
#ifndef NUM_LOD
	#define NUM_LOD 3
#endif
#ifndef __LOD_TYPES
#define __LOD_TYPES 1
	enum LOD_TYPE { LOD_HI, LOD_ME, LOD_LO };
#endif

//Global or Local position texture:
#define GLOBAL_POS_TEXTURE
#undef LOCAL_POS_TEXTURE

#define INIT_WINDOW_WIDTH		1920
#define INIT_WINDOW_HEIGHT		1080

//#define INIT_WINDOW_WIDTH		720
//#define INIT_WINDOW_HEIGHT		480

//=======================================================================================

/* This will expose:
 * M_E        - e
 * M_LOG2E    - log2(e)
 * M_LOG10E   - log10(e)
 * M_LN2      - ln(2)
 * M_LN10     - ln(10)
 * M_PI       - pi
 * M_PI_2     - pi/2
 * M_PI_4     - pi/4
 * M_1_PI     - 1/pi
 * M_2_PI     - 2/pi
 * M_2_SQRTPI - 2/sqrt(pi)
 * M_SQRT2    - sqrt(2)
 * M_SQRT1_2  - 1/sqrt(2)
 */
#ifndef _USE_MATH_DEFINES
	#define _USE_MATH_DEFINES	1
#endif

#include <math.h>

//=======================================================================================

#ifndef DEG2RAD
	#define DEG2RAD	0.01745329251994329576f
#endif

//=======================================================================================

#ifndef RAD2DEG
	#define RAD2DEG	57.29577951308232087679f
#endif

//=======================================================================================

// For cScreenText.cpp
#ifndef SCREEN_TEXT_BUFFER_SIZE
	#define SCREEN_TEXT_BUFFER_SIZE	1024
#endif

//=======================================================================================

#ifndef FREE_TEXTURE
	#define FREE_TEXTURE( ptr )				\
	if( ptr )								\
	{										\
		glDeleteTextures( 1, &ptr );		\
		ptr = 0;							\
	}
#endif

//=======================================================================================

#ifndef FREE_INSTANCE
	#define FREE_INSTANCE( ptr )			\
	if( ptr )								\
	{										\
		delete ptr;							\
		ptr = 0;							\
	}
#endif

//=======================================================================================

#ifndef BYTE2KB
	#define BYTE2KB( b ) b / 1024
#endif

//=======================================================================================

#ifndef BYTE2MB
	#define BYTE2MB( b ) b / 1048576
#endif

//=======================================================================================

#ifndef BYTE2GB
	#define BYTE2GB( b ) b / 1073741824
#endif

//=======================================================================================


// For cXmlParser.cpp:
#ifdef __unix
	#ifndef TIXML_USE_STL
		#define TIXML_USE_STL 1
	#endif
#endif
#define NUM_INDENTS_PER_SPACE 2;

//=======================================================================================

// For cGlslManager.cpp:
#ifndef SHADER_OBJECT_MAX_VARIABLES
    #define SHADER_OBJECT_MAX_VARIABLES 64
#endif

#ifndef USE_GEOMETRY_SHADERS
    #define USE_GEOMETRY_SHADERS		1
#endif

#ifndef USE_GI_MODELS
    #define USE_GI_MODELS				0
#endif

#ifndef USE_INSTANCING
    #define USE_INSTANCING				1
#endif

#ifndef STRING_UTILS_BUFFER_SIZE
    #define STRING_UTILS_BUFFER_SIZE	65536
#endif

//=======================================================================================

#ifndef __XYZ
#define __XYZ 1
typedef struct {
   double x,y,z;
} XYZ;
#endif

#ifndef __PIXELA
#define __PIXELA 1
typedef struct {
   unsigned char r,g,b,a;
} PIXELA;
#endif

#ifndef __COLOUR
#define __COLOUR 1
typedef struct {
   double r,g,b;
} COLOUR;
#endif

#define CROSSPROD( p1, p2, p3 ) \
   p3.x = p1.y*p2.z - p1.z*p2.y; \
   p3.y = p1.z*p2.x - p1.x*p2.z; \
   p3.z = p1.x*p2.y - p1.y*p2.x

#ifndef __NORMALIZE
#define __NORMALIZE 1
#define NORMALIZE(p,length) \
   length = sqrt(p.x * p.x + p.y * p.y + p.z * p.z); \
   if( length != 0 ) { \
      p.x /= length; \
      p.y /= length; \
      p.z /= length; \
   } else { \
      p.x = 0; \
      p.y = 0; \
      p.z = 0; \
   }
#endif

#ifndef LOD_STRUCT
#define LOD_STRUCT

struct sVBOLod
{
	unsigned int id;
	unsigned int primitivesWritten;
	unsigned int primitivesGenerated;
};

#endif

#ifndef FREE_MEMORY
#define FREE_MEMORY(ptr)	\
    if (ptr) {				\
		delete ptr;			\
        ptr=0;			\
    }
#endif

#ifndef FREE_ARRAY
#define FREE_ARRAY(ptr)	\
    if (ptr) {				\
		delete [] ptr;			\
        ptr=0;			\
    }
#endif

#ifndef FREE_VBO
#define FREE_VBO(ptr) \
	if (ptr) { \
		glDeleteBuffers(1,&ptr); \
		ptr = 0; \
	}
#endif

#ifndef FREE_TEXTURE
#define FREE_TEXTURE(ptr) \
	if (ptr) { \
		glDeleteTextures(1,&ptr); \
		ptr = 0; \
	}
#endif

#ifndef FREE_OGL_LIST
#define FREE_OGL_LIST(ptr) \
	if (ptr) { \
		glDeleteLists(ptr,1); \
		ptr = 0; \
	}
#endif

//=======================================================================================

// For cVboManager.cpp:
#ifndef MAX_INSTANCES
	#define MAX_INSTANCES 4096
#endif

//=======================================================================================

// For cCharacterModel.cpp:
#ifndef PARTS_PER_MODEL
	#define PARTS_PER_MODEL	3
	enum MODEL_PART
	{
		MP_HEAD,
		MP_TORSO,
		MP_LEGS
	};
#endif

#ifndef GENDERS_PER_MODEL
	#define GENDERS_PER_MODEL 2
	enum MODEL_GENDER
	{
		MG_MALE,
		MG_FEMALE
	};
#endif

#ifndef TYPES_PER_MODEL
	#define TYPES_PER_MODEL 2
    enum MODEL_TYPE
	{
		MT_HUMAN,
		MT_LEMMING
	};
#endif

//=======================================================================================

// For cScenario.cpp:
// For cModel3D.cpp:
#ifndef _MODEL_MESH
	#define _MODEL_MESH
	typedef struct MODEL_MESH
	{
		unsigned int	vbo_index;
		unsigned int	vbo_frame;
		unsigned int	vbo_size;
		unsigned int	vbo_id;
		unsigned int	tex_id;
		float			tint[3];
		float			ambient[3];
		float			opacity;
		bool			tint_set;
		bool			ambient_set;
		std::string		tex_file;
	} model_mesh;

	#define INIT_MODEL_MESH( mm )		\
		mm.tex_id		= 0;			\
		mm.vbo_id		= 0;			\
		mm.tint[0]		= 0.0f;			\
		mm.tint[1]		= 0.0f;			\
		mm.tint[2]		= 0.0f;			\
		mm.opacity		= 0.0f;			\
		mm.ambient[0]	= 0.0f;			\
		mm.ambient[1]	= 0.0f;			\
		mm.ambient[2]	= 0.0f;			\
		mm.tint_set		= false;		\
		mm.ambient_set	= false

#endif

//=======================================================================================

#ifndef MODEL_PROPS_TYPES
	#define MODEL_PROPS_TYPES 3
	enum MODEL_PROPS_TYPE
	{
		MPT_CLOTHING,
		MPT_FACIAL,
		MPT_RIGGING
	};
#endif

//=======================================================================================

// For cModelProps.cpp:
#ifndef PARTS_PER_PROP
	#define PARTS_PER_PROP	17
	enum PROP_PART
	{
		PP_HEAD,
		PP_HAIR,
		PP_TORSO,
		PP_LEGS,
		PP_ATLAS,
		PP_WRINKLES,
		PP_PATTERN,

		PP_FACIAL_WRINKLES,
		PP_FACIAL_EYE_SOCKETS,
		PP_FACIAL_SPOTS,
		PP_FACIAL_BEARD,
		PP_FACIAL_MOUSTACHE,
		PP_FACIAL_MAKEUP,

		PP_RIGGING_ZONES,
		PP_RIGGING_WEIGHTS,
		PP_RIGGING_DISPLACEMENT,
		PP_RIGGING_ANIMATION
	};
#endif

#ifndef SUBTYPES_PER_PROP
#define SUBTYPES_PER_PROP	9
	enum PROP_SUBTYPE
	{
		PST_SKIN,
		PST_HAIR,
		PST_CAP,
		PST_HEAD,
		PST_TORSO,
		PST_LEGS,
		PST_TORSO_AND_LEGS,
		PST_FACE,
		PST_RIG
	};
#endif

//=======================================================================================

// For cCrowdManager.cpp:
#ifndef __GROUP_FORMATIONS
#define __GROUP_FORMATIONS
	enum GROUP_FORMATION
	{
		GFRM_TRIANGLE,
		GFRM_SQUARE,
		GFRM_HEXAGON,
		GFRM_CIRCLE
	};
#endif

//=======================================================================================

#ifndef __MDP_TYPE
#define __MDP_TYPE
	enum MDP_TYPE
	{
		MDPT_SQUARE,
		MDPT_HEXAGON
	};
	enum MDP_MACHINE_STATE
	{
		MDPS_IDLE,
		MDPS_INIT_STRUCTURES_ON_HOST,
		MDPS_INIT_PERMS_ON_DEVICE,
		MDPS_UPLOADING_TO_DEVICE,
		MDPS_ITERATING_ON_DEVICE,
		MDPS_DOWNLOADING_TO_HOST,
		MDPS_UPDATING_POLICY,
		MDPS_READY,
		MDPS_ERROR
	};
#endif

//=======================================================================================

