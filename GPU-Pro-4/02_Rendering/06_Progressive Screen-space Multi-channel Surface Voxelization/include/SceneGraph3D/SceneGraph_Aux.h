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

#ifndef _SCENE_GRAPH_AUX_
#define _SCENE_GRAPH_AUX_

#include "Vector2D.h"
#include "Vector3D.h"
#include "Vector4D.h"

#ifdef WIN32
    #define GET_TIME() (timeGetTime()/1000.0f)
#else
    #define GET_TIME() clock()/(float)CLOCKS_PER_SEC
#endif

int parseVec4(float &s, Vector3D &val, char * text);
int parseVec3(Vector3D &val, char * text);
int parseVec2(Vector2D &val, char * text);
int parseBoolean(bool &val, char * text);
int parseFloat(float &val, char * text);
int parseShort(short &val, char * text);
int parseUShort(unsigned short &val, char * text);
int parseInteger(int &val, char * text);
int parseUInteger(unsigned int &val, char * text);
int parseIVec (vector <int> &vec, const string& text);
int parseStringsVector (vector <string> &vec, const string& text);

char *skipParameterName(char *buf);

#define SAFEFREE(_x)        { if ((_x) != NULL) { free    ((_x)); _x = NULL; } }
#define SAFEDELETE(_x)      { if ((_x) != NULL) { delete   (_x);  _x = NULL; } }
#define SAFEDELETEARRAY(_x) { if ((_x) != NULL) { delete[] (_x);  _x = NULL; } }

class Node3D * buildNode(char *name);

int exp2i (int val);
int getPowerOfTwo (int val);

#endif

