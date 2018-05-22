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

#include <stdio.h>
#include <sstream>

#ifndef WIN32
	#include <ctype.h>
#else
	#pragma warning (disable : 4996)
#endif

#include "SceneGraph.h"

using namespace std;

int parseVec2(Vector2D &vec, char * text)
{
	if (!text)
		return SCENE_GRAPH_ERROR_PARSING;
	if ( sscanf(text, "%f%*[ ,\t]%f", &(vec.x), &(vec.y)) < 2 )
		return SCENE_GRAPH_ERROR_PARSING;
	else
		return SCENE_GRAPH_ERROR_NONE;
}

int parseVec3(Vector3D &vec, char * text)
{
	if (!text)
		return SCENE_GRAPH_ERROR_PARSING;
	if ( sscanf(text, "%f%*[ ,\t]%f%*[ ,\t]%f", &(vec.x), &(vec.y), &(vec.z)) < 3 )
		return SCENE_GRAPH_ERROR_PARSING;
	else
		return SCENE_GRAPH_ERROR_NONE;
}

int parseVec4(float &s, Vector3D &vec, char * text)
{
	if (!text)
		return SCENE_GRAPH_ERROR_PARSING;
	if ( sscanf(text, "%f%*[ ,\t]%f%*[ ,\t]%f%*[ ,\t]%f", &(s), &(vec.x), &(vec.y), &(vec.z)) < 4 )
		return SCENE_GRAPH_ERROR_PARSING;
	else
		return SCENE_GRAPH_ERROR_NONE;
}

int parseFloat(float &val, char * text)
{
	if (!text)
		return SCENE_GRAPH_ERROR_PARSING;
	if ( sscanf(text, "%f", &(val)) < 1 )
		return SCENE_GRAPH_ERROR_PARSING;
	else
		return SCENE_GRAPH_ERROR_NONE;
}

int parseShort(short &val, char * text)
{
	if (!text)
		return SCENE_GRAPH_ERROR_PARSING;
	if ( sscanf(text, "%hd", &(val)) < 1 )
		return SCENE_GRAPH_ERROR_PARSING;
	else
		return SCENE_GRAPH_ERROR_NONE;
}

int parseUShort(unsigned short &val, char * text)
{
	if (!text)
		return SCENE_GRAPH_ERROR_PARSING;
	if ( sscanf(text, "%hd", &(val)) < 1 )
		return SCENE_GRAPH_ERROR_PARSING;
	else
		return SCENE_GRAPH_ERROR_NONE;
}

int parseInteger(int &val, char * text)
{
	if (!text)
		return SCENE_GRAPH_ERROR_PARSING;
	if ( sscanf(text, "%d", &(val)) < 1 )
		return SCENE_GRAPH_ERROR_PARSING;
	else
		return SCENE_GRAPH_ERROR_NONE;
}

int parseUInteger(unsigned int &val, char * text)
{
	if (!text)
		return SCENE_GRAPH_ERROR_PARSING;
	if ( sscanf(text, "%d", &(val)) < 1 )
		return SCENE_GRAPH_ERROR_PARSING;
	else
		return SCENE_GRAPH_ERROR_NONE;
}

int parseIVec (vector <int> &vec, const string& text)
{
    string  temp, str = text;
    int     tempi = 0;
    size_t  pos;

    // does the string have a space a comma or a tab in it?
    // store the position of the delimiter
    while ((pos = str.find_first_of (" ,\t", 0)) != string::npos)
    {
        temp = str.substr (0, pos); // get the token
        stringstream ss (temp);
        ss >> tempi;                // check if the token is an integer
        if (ss.fail ())
            return SCENE_GRAPH_ERROR_PARSING;
        vec.push_back (tempi);      // and put it into the array
        str.erase (0, pos + 1);     // erase it from the source
    }

    stringstream ss (str);
    ss >> tempi;
    if (ss.fail ())
        return SCENE_GRAPH_ERROR_PARSING;
    vec.push_back (tempi);          // the last token is all alone

    return SCENE_GRAPH_ERROR_NONE;
}

int parseBoolean(bool &val, char * text)
{
	if (!text)
		return SCENE_GRAPH_ERROR_PARSING;

	if (STR_EQUAL(text,"off"))
		val=false;
	else if (STR_EQUAL(text,"on"))
		val=true;
	else if (STR_EQUAL(text,"true"))
		val=true;
	else if (STR_EQUAL(text,"false"))
		val=false;
	else if (STR_EQUAL(text,"1"))
		val=true;
	else if (STR_EQUAL(text,"0"))
		val=false;
	else if (STR_EQUAL(text,"yes"))
		val=true;
	else if (STR_EQUAL(text,"no"))
		val=false;
	return SCENE_GRAPH_ERROR_NONE;
}

int parseStringsVector (vector <string> &vec, const string& text)
{
    string  temp, str = text;
    size_t  pos;

    // does the string have a space a comma or a tab in it?
    // store the position of the delimiter
    while ((pos = str.find_first_of (" ,\t", 0)) != string::npos)
    {
        vec.push_back (str.substr (0, pos));    // get the token and put it into the array
        str.erase (0, pos + 1);     // erase it from the source
    }

    vec.push_back (str);            // the last token is all alone

    return SCENE_GRAPH_ERROR_NONE;
}

char *skipParameterName(char *buf)
{
	char * first = buf, * found;
	int next;
	while (first==(char*)'\n' || first==(char*)' ' || first == (char*)'\t')
		first++;
	next = strcspn(first," \t\n");
	found = first+next;
	return found;
}

