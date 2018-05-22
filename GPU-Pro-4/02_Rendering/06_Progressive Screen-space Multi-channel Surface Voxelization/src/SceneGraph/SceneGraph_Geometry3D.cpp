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

#include "SceneGraph.h"

#include "raw.h"
#include "obj.h"

Geometry3D::Geometry3D()
{
	mesh = NULL;
}

Geometry3D::~Geometry3D()
{
	SAFEDELETE (mesh);
}

void Geometry3D::app()
{
	if (!active)
		return;

	if (getDirty ())
	{
		if (mesh)
#ifdef BUFFER_OBJECT
		    mesh->compileBuffer ();
#else
		    mesh->compileLists ();
#endif
	
		setDirty (false);
	}
	
	Node3D::app();
}

void Geometry3D::cull()
{
	Node3D::cull();
}

void Geometry3D::draw()
{
	if (!isVisible())
		return;

	if (mesh)
	{
		switch (render_mode)
		{
		case SCENE_GRAPH_RENDER_MODE_NORMAL:
		    mesh->drawOpaque();
		    break;
		case SCENE_GRAPH_RENDER_MODE_TRANSPARENCY:
		    mesh->drawTransparent();
		    break;
		case SCENE_GRAPH_RENDER_MODE_HIDDEN:
		default:
		    // do nothing
		    break;
		}
	}
}

void Geometry3D::parse(xmlNodePtr pXMLNode)
{
	char * val = NULL;

	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"file");
	if (val)
	{
		char * path = world->getFullPath(val);
		if (path)
			load(path);
		else
			EAZD_TRACE ("Geometry3D::parse() : ERROR - File \"" << val << "\" is corrupt or does not exist.");
		free (path);
		xmlFree(val);
	}
	val = (char *)xmlGetProp(pXMLNode, (xmlChar *)"doublesided");
	if (val)
	{
		bool ds=false;
		parseBoolean(ds, val);
		mesh->setDoubleSided(ds);
		setDirty(true);
		xmlFree(val);
	}
	
	Node3D::parse(pXMLNode);
}

void Geometry3D::load(char * file)
{
    EAZD_ASSERTALWAYS (file);

	if (mesh)
		delete mesh;
	
	EAZD_PRINT ("Geometry3D::load() : INFO - Loading mesh \"" << file << "\"");
	mesh = new Mesh3D(file,world->getPaths());
	EAZD_ASSERTALWAYS (mesh);
	EAZD_PRINT ("Geometry3D::load() : INFO - Mesh " << mesh->getName () << " has: " << mesh->numfaces << " faces, " << mesh->numvertices << " vertices");

	// mesh bbox; not world bbox
	EAZD_PRINT ("Geometry3D::load() : INFO - Mesh Bounding Box: ");
	getBBox().dump();
	
	setDirty (false);
}

Vector3D Geometry3D::getWorldPosition()
{
	Matrix4D m = getTransform();
	return mesh->getCenter()*m;
}

void Geometry3D::processMessage(char * msg)
{
    if (STR_EQUAL(msg,"dump"))
    {
        if (mesh)
            mesh->dump();
    }
    else
    	Node3D::processMessage(msg);
}

void Geometry3D::init()
{
	Node3D::init();
}

BBox3D Geometry3D::getBBox()
{
	return mesh->getBoundingBox();
}

BSphere3D Geometry3D::getBSphere()
{
	return mesh->getBoundingSphere();
}

float * Geometry3D::flattenRawTriangles(long *number)
{
	long i,j,index=0;
	long numgroups = mesh->numgroups;
	float * vertices = (float *)malloc(9*mesh->numfaces*sizeof(float));
	for (i=0;i<numgroups;i++)
	{
		long numgroupfaces = mesh->groups[i].numfaces;
		for (j=0;j<numgroupfaces;j++)
		{
			vertices[9*index+0] = mesh->vertices[mesh->groups[i].faces[j].vertIdx[0]].x;
			vertices[9*index+1] = mesh->vertices[mesh->groups[i].faces[j].vertIdx[0]].y;
			vertices[9*index+2] = mesh->vertices[mesh->groups[i].faces[j].vertIdx[0]].z;
			vertices[9*index+3] = mesh->vertices[mesh->groups[i].faces[j].vertIdx[1]].x;
			vertices[9*index+4] = mesh->vertices[mesh->groups[i].faces[j].vertIdx[1]].y;
			vertices[9*index+5] = mesh->vertices[mesh->groups[i].faces[j].vertIdx[1]].z;
			vertices[9*index+6] = mesh->vertices[mesh->groups[i].faces[j].vertIdx[2]].x;
			vertices[9*index+7] = mesh->vertices[mesh->groups[i].faces[j].vertIdx[2]].y;
			vertices[9*index+8] = mesh->vertices[mesh->groups[i].faces[j].vertIdx[2]].z;
			index++;
		}
	}
	*number=index;
	return vertices;
}
