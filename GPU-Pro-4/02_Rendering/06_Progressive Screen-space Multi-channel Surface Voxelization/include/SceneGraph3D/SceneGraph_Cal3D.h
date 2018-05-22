//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Scene Graph 3D                                                          //
//  Athanasios Gaitatzes (gaitat at yahoo dot com), 2009                    //
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

#ifndef _SCENE_GRAPH_CAL3D_
#define _SCENE_GRAPH_CAL3D_

#include <libxml/tree.h>

#include "SceneGraph_Node3D.h"
#include "BSphere3D.h"

#include "cal3d/cal3d.h"

class Cal3D : public Node3D
{
public:
    Cal3D (void);
    virtual ~ Cal3D (void);
    virtual void parse (xmlNodePtr pXMLnode);
    virtual void load (const string & file);
    virtual void init ();
    virtual void app  ();
    virtual void draw ();

    void createDataArrays (void);   // create the mesh data

    inline BSphere3D getBSphere (void) { return bsphere_; }
    void calcBSphere (void);        // computes bounding sphere

    static void registerEvent(char *evt) { eventmap.push_back(evt); }

private:
    char          * path_;
    CalCoreModel  * calCoreModel_;
    CalModel      * calModel_;

    float        (* meshVertices_)[3];
    float        (* meshNormals_)[3];
    float        (* meshTextureCoordinates_)[2];
    CalIndex     (* meshFaces_)[3];

    int             animationCycle_[12], totalCycles_,
                    animationAction_[12], totalActions_,
                    previousCycle_;
    bool            pause_;
    float           last_time_;
    BSphere3D       bsphere_;
    bool            alpha_;

	static int      new_update_offset_;  
	int				update_offset_, update_;     
	
    void drawSkeleton (void);   // for debugging
    void drawBBox (void);       // for debugging
    void drawGrid (void);       // for debugging

    static vector <char *> eventmap;
    virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
};

#endif

