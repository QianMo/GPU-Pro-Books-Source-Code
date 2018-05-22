//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Scene Graph 3D                                                          //
//  Georgios Papaioannou, 2009                                              //
//                                                                          //
//  This is a free, extensible scene graph management library that works    //
//  along with the EaZD deferred renderer. Both libraries and their source  //
//  code are free. If you use this code as is or any part of it in any kind //
//  of project or product, please acknowledge the source and its author.    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#ifndef _SCENE_GRAPH_3D_
#define _SCENE_GRAPH_3D_

#define SCENE_GRAPH_VERSION 0
#define SCENE_GRAPH_REVISION 1

#define _USING_DEFERRED_RENDERING_

#define SCENE_GRAPH_ERROR_NONE            0
#define SCENE_GRAPH_ERROR_PARSING         1
#define SCENE_GRAPH_ERROR_NULL_PARENT     2
#define SCENE_GRAPH_ERROR_NULL_CHILD      3
#define SCENE_GRAPH_ERROR_DEVICE_INIT     4
#define SCENE_GRAPH_ERROR_DIRECTINPUT 	  5

#define SCENE_GRAPH_SHADOWS_DISK		  3

#define SCENE_GRAPH_USER_MODE_NAVIGATE    0
#define SCENE_GRAPH_USER_MODE_ORBIT       1
#define SCENE_GRAPH_USER_MODE_PASSIVE     2

#define SCENE_GRAPH_RENDER_MODE_HIDDEN         0
#define SCENE_GRAPH_RENDER_MODE_NORMAL         1
#define SCENE_GRAPH_RENDER_MODE_TRANSPARENCY   2
#define SCENE_GRAPH_RENDER_MODE_MEDIA          3
#define SCENE_GRAPH_RENDER_MODE_PARTICLES      4

#ifdef _USING_DEFERRED_RENDERING_
	#include "DeferredRenderer.h"
#else
	#include <GL/gl.h>
#endif

#include "SceneGraph_EventMessage.h"
#include "SceneGraph_Node3D.h"
#include "SceneGraph_Timer3D.h"
#include "SceneGraph_Input3D.h"
#include "SceneGraph_Group3D.h"
#include "SceneGraph_TransformationNode3D.h"
#include "SceneGraph_LinearMotion3D.h"
#include "SceneGraph_Spinner3D.h"
#include "SceneGraph_Geometry3D.h"
#include "SceneGraph_Camera3D.h"
#include "SceneGraph_User3D.h"
#include "SceneGraph_World3D.h"
#include "SceneGraph_Light3D.h"
#include "SceneGraph_Aux.h"
#include "SceneGraph_Cal3D.h"
#include "SceneGraph_Force3D.h"
#endif
