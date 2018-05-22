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

#ifdef WIN32
    #include <windows.h>
#endif

#include <time.h>
#include <iostream>
#include <fstream>

#include "SceneGraph.h"
// #include <libxml/xinclude.h>

using namespace std;

World3D::World3D()
{
	dr = NULL;
	err_code = SCENE_GRAPH_ERROR_NONE;
	debug = false;
	app_time = 0.0;
	dt = 0.0;
	parent = NULL;
	world = this;
	init_ambient = Vector3D(0.2f,0.2f,0.2f);
	doc = NULL;
	paths.push_back((char *) ".");
	active_camera = NULL;
	
#ifdef _USING_DEFERRED_RENDERING_
	init_shadows = SCENE_GRAPH_SHADOWS_DISK;
	init_hdr_mode = DR_HDR_MANUAL;
	init_hdr_white = Vector3D(1,1,1);
	init_hdr_key = 0.5f;
#endif

	SceneGraphRegisterEvents();
	TextureManager3D::init();
}

World3D::~World3D()
{
	if (doc)
		xmlFreeDoc(doc);

    // Free the global variables that may have been allocated by the parser.
    xmlCleanupParser ();

	paths.clear();

	for (int i=gi_light_names.size()-1; i>=0; i--)
	{
		char * str = gi_light_names[i];
		if (str)
			free (str);
	}
	gi_light_names.clear();
}

void World3D::app()
{
	processMessages();
	
	double cur_time = GET_TIME()-init_time;
	dt = cur_time - app_time;
	app_time = cur_time;
	Group3D::app();

	dispatchMessages();
}

void World3D::cull()
{
	Group3D::cull();
}

void World3D::clear(long mask)
{
	glClear(mask);
}

void World3D::draw()
{
	Group3D::draw();
}

void World3D::init()
{
	ambient = init_ambient;
	hdr_white = init_hdr_white;
	hdr_key = init_hdr_key;
	hdr_mode = init_hdr_mode;
	if (dr)
	{
		dr->setSceneRoot(this);
		dr->setAmbient(init_ambient.x, init_ambient.y, init_ambient.z);
#ifdef _USING_DEFERRED_RENDERING_
		dr->setShadowMethod(init_shadows);
		dr->setHDRKey(hdr_key);
		dr->setHDRWhitePoint(hdr_white.x,hdr_white.y,hdr_white.z);
		dr->setHDRMethod(hdr_mode);
#endif
	}
	init_time = GET_TIME();
	dt = 0.0001;
	Group3D::init();
	check();

	EAZD_PRINT ("World3D::init() : INFO - Scene Bounding Box: ");
	getBBox().dump();

// #ifdef NVIDIA
#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

	GLint total_mem_kb = 0;
	glGetIntegerv (GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);

	GLint cur_avail_mem_kb = 0;
	glGetIntegerv (GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);
// #endif
#ifdef ATI
	UINT n = wglGetGPUIDsAMD (0, 0);
	UINT *ids = new UINT[n];
	size_t total_mem_mb = 0;
	wglGetGPUIDsAMD (n, ids);
	wglGetGPUInfoAMD (ids[0], WGL_GPU_RAM_AMD, GL_UNSIGNED_INT, sizeof(size_t), &total_mem_mb);
#endif

	EAZD_PRINT ("World3D::init() : OpenGL info - Currently available GPU memory: " << cur_avail_mem_kb / 1024.0f << "MB");

#ifdef _USING_DEFERRED_RENDERING_
	for (unsigned int i=0; i<gi_light_names.size(); i++)
	{
		Node3D * nd = getNodeByName(gi_light_names[i]);
		Light3D *l=NULL;
		if (nd)
			l = dynamic_cast<Light3D*>(nd);
		if (l)
			dr->getLight(l->getID())->enableGI(true);
	}

	dr->initLighting();
#endif

	EVENT_OCCURED("init");
}

void World3D::reset()
{
	ambient = init_ambient;
	hdr_white = init_hdr_white;
	hdr_key = init_hdr_key;
	init_time = GET_TIME();
	dt = 0.0001;
	Group3D::reset();
	if (!dr)
		return;
#ifdef _USING_DEFERRED_RENDERING_
	dr->setShadowMethod(init_shadows);
	dr->setHDRKey(hdr_key);
	dr->setHDRWhitePoint(hdr_white.x,hdr_white.y,hdr_white.z);
#endif
	EVENT_OCCURED("init");
	app();
}

void World3D::parse(xmlNodePtr pXMLNode)
{
	char * val = NULL;
	
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"debug");
	if (val)
	{
		parseBoolean(debug, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"ambient");
	if (val)
	{
		parseVec3(init_ambient, val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"background");
	if (val)
	{
		Vector3D bkg;
		parseVec3(bkg, val);
		xmlFree(val);
		dr->setBackground(bkg.x,bkg.y,bkg.z);
	}
#ifdef _USING_DEFERRED_RENDERING_
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"shadows");
	if (val)
	{
	    init_shadows = DR_SHADOW_GAUSS;

		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"hdr_mode");
	if (val)
	{
	    if (STR_EQUAL(val,"manual"))
			init_hdr_mode = DR_HDR_MANUAL;
	    else if (STR_EQUAL(val,"auto"))
		    init_hdr_mode = DR_HDR_AUTO;
	    xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"hdr_key");
	if (val)
	{
		parseFloat(init_hdr_key,val);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"hdr_white");
	if (val)
	{
		parseVec3(init_hdr_white,val);
		xmlFree(val);
	}
#endif

	xmlNodePtr cur = pXMLNode->xmlChildrenNode;
	while (cur != NULL)
	{
	    if (STR_EQUAL((char*)cur->name,"directory"))
		{
			val = (char *)xmlGetProp(cur, (xmlChar *)"path");
			if (val)
			{
				paths.push_back(STR_DUP(val));
				xmlFree(val);
			}
		}
		else if (STR_EQUAL((char*)cur->name,"gi"))
		{
			val = (char *)xmlGetProp(cur, (xmlChar *)"contribution");
			if (val)
			{
				float con=0;
				parseFloat(con,val);
				if (con>0)
					dr->getGIRenderer()->setFactor(con);
				xmlFree(val);
			}
			val = (char *)xmlGetProp(cur, (xmlChar *)"bounces");
			if (val)
			{
				int bou=0;
				parseInteger(bou,val);
				if (bou>=0)
					dr->getGIRenderer()->setBounces(bou);
				xmlFree(val);
			}
			val = (char *)xmlGetProp(cur, (xmlChar *)"params");
			if (val)
			{
				dr->getGIRenderer()->setParamString(val);
				xmlFree(val);
			}
			val = (char *)xmlGetProp(cur, (xmlChar *)"active");
			if (val)
			{
				bool en=0;
				parseBoolean(en,val);
				dr->enableGI(en);
				xmlFree(val);
			}
			val = (char *)xmlGetProp(cur, (xmlChar *)"range");
			if (val)
			{
				float rg=0;
				parseFloat(rg,val);
				if (rg>0)
					dr->getGIRenderer()->setRange(rg);
				xmlFree(val);
			}
			val = (char *)xmlGetProp(cur, (xmlChar *)"factor");
			if (val)
			{
				float f=0;
				parseFloat(f,val);
				if (f>0)
					dr->getGIRenderer()->setFactor(f);
				xmlFree(val);
			}
			val = (char *)xmlGetProp(cur, (xmlChar *)"resolution");
			if (val)
			{
				int res=0;
				parseInteger(res,val);
				if (res>4)
					dr->setVolumeBufferResolution(res);
				xmlFree(val);
			}
			val = (char *)xmlGetProp(cur, (xmlChar *)"samples");
			if (val)
			{
				int samples=0;
				parseInteger(samples,val);
				if (samples>0)
					dr->getGIRenderer()->setNumSamples(samples);
				xmlFree(val);
			}
			val = (char *)xmlGetProp(cur, (xmlChar *)"lights");
			if (val)
			{
				char * tok = strtok(val," ,\t");
				while (tok)
				{
					gi_light_names.push_back(strdup(tok));
					tok = strtok(NULL," ,\t");
				}
				xmlFree(val);
			}
		}
		cur = cur->next;
	}

	Group3D::parse(pXMLNode);
}

void World3D::load(char *scenefile)
{
	err_code = SCENE_GRAPH_ERROR_NONE;
	strcpy (worldfile, scenefile);
	
	if (!scenefile)
	{
		err_code = SCENE_GRAPH_ERROR_PARSING;
		return;
	}
	FILE* fp = fopen(scenefile,"rt");
	if (!fp)
	{
		err_code = SCENE_GRAPH_ERROR_PARSING;
		return;
	}
	fclose(fp);
    doc = xmlParseFile (scenefile);
    if (!doc)
    {
		err_code = SCENE_GRAPH_ERROR_PARSING;
		return;
    }
#if 0
    if (xmlXIncludeProcess(doc) <= 0)
    {
		err_code = SCENE_GRAPH_ERROR_PARSING;
		return;
    }
#endif
    xml_root = xmlDocGetRootElement (doc);
    if (!xml_root)
    {
		err_code = SCENE_GRAPH_ERROR_PARSING;
		return;
    }

	parse(xml_root);
}

char * World3D::getFullPath(char * file)
{
	char * final = NULL;
	for (unsigned int i=0; i<paths.size();i++)
	{
		char * dir = paths.at(i);
		int len = strlen(dir)+strlen(file)+2; // 1 for delimiter & 1 for null
		final = (char *)malloc(len*sizeof(char));
		sprintf(final,"%s%c%s",dir,DIR_DELIMITER,file);
		ifstream test;
		test.open(final);
		if (!test.fail())
		{
			test.close();
			return final;
		}
		free (final);
	}
	return NULL;
}

void World3D::check()
{
	Node3D* nd;
	// check for at least one camera --> create default if none
	nd = getNodeByType((char *) "Camera3D");
	if (nd==NULL)
	{
		Camera3D *cam = new Camera3D();
		addChild(cam);
		cam->setParent(this);
		cam->setWorld(this);
		active_camera = cam;
	}
	// check for at least one user --> create default if none
	nd = getNodeByType((char *) "User3D");
	if (nd==NULL)
	{
		User3D *usr = new User3D();
		addChild(usr);
		usr->setParent(this);
		usr->setWorld(this);
	}
	// ensure exactly one primary camera
	if (active_camera==NULL)
	{
		nd = getNodeByType((char *) "Camera3D");
		active_camera = dynamic_cast<Camera3D*>(nd);
	}
	// check that camera is attached to something
	// attach camera to user otherwise
	if ((active_camera->attachedTo()==NULL) &&
        (dynamic_cast<TransformNode3D*>(active_camera->getParent())==NULL))
	{
		nd = getNodeByType((char *) "User3D");
		active_camera->attachTo(dynamic_cast<User3D*>(nd));
	}
}

void World3D::processMessage(char * msg)
{
	if (STR_EQUAL(msg,"reset"))
		reset();
#ifdef _USING_DEFERRED_RENDERING_
	else if (SUBSTR_EQUAL(msg,"ambient"))
	{
		parseVec3(ambient,skipParameterName(msg));
		dr->setAmbient(ambient.x, ambient.y, ambient.z);
	}
	else if (SUBSTR_EQUAL(msg,"hdr_white"))
	{
		parseVec3(hdr_white,skipParameterName(msg));
		dr->setHDRWhitePoint(hdr_white.x, hdr_white.y, hdr_white.z);
	}
	else if (SUBSTR_EQUAL(msg,"hdr_key"))
	{
		parseFloat(hdr_key,skipParameterName(msg));
		dr->setHDRKey(hdr_key);
	}
	else if (SUBSTR_EQUAL(msg,"background"))
	{
		Vector3D bkg;
		parseVec3(bkg, skipParameterName(msg));
		dr->setBackground(bkg.x,bkg.y,bkg.z);
	}
#endif
	else
		Group3D::processMessage(msg);
}
