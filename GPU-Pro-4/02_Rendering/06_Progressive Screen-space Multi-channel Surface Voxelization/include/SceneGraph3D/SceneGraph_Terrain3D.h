
#ifndef _SCENE_GRAPH_TERRAIN3D_
#define _SCENE_GRAPH_TERRAIN3D_

#include <vector>

#include "Grass3D/HeightMap3D.h"
#include "Grass3D/Portal3D.h"
#include "Grass3D/DistanceObject.h"

#include "Vector4D.h"
#include "Frustum3D.h"

#include "SceneGraph_Camera3D.h"

typedef DistanceObject <Portal3D *> LPPortal;
typedef vector <LPPortal> LPPortalManager;

class Terrain3D : public Node3D
{
public:
    Terrain3D ();
   ~Terrain3D ();

    virtual void parse (xmlNodePtr pXMLnode);
    virtual void init (void);
    virtual void draw ();
    void         draw (Frustum3D & frustum, float elapsedTime);
    void         drawSimple (float elapsedTime);

    void    adjustCameraYLocation (Camera3D & camera, float cameraRadius);

    void    setOcclusionCullingFlag (bool occCull)  { occlusionCullCells_ = occCull;
                                                      resetRenderingRoutine ();     }
    void    setAlphaToCoverageFlag (bool alpha)     { alphaToCoverage_    = alpha;  }
    void    setFrustumCullingFlag (bool frCull)     { frustumCullCells_   = frCull;
                                                      resetRenderingRoutine ();     }
    void    setWireFrameFlag (bool wFrame)          { drawWireFrame_      = wFrame; }
    void    setDrawGrassFlag (bool dGrass)          { drawGrass_          = dGrass; }

    void    setAlphaReference (float ref)           { alphaReference_     = ref;    }
    void    setAlphaBooster (float boost)           { alphaBooster_       = boost;  }

    const float getAlphaReference ()                { return alphaReference_;   }
    const float getAlphaBooster ()                  { return alphaBooster_;     }

    const bool  getOcclusionCullingFlag ();

    const bool  getAlphaToCoverageFlag ()           { return alphaToCoverage_;  }
    const bool  getFrustumCullingFlag ()            { return frustumCullCells_; }
    const bool  getWireFrameFlag ()                 { return drawWireFrame_;    }
    const bool  getDrawGrassFlag ()                 { return drawGrass_;        }


    static void registerEvent(char *evt) { eventmap.push_back(evt); }

private:
    void  resetRenderingRoutine ();
    bool  appendToGreenChannel (Texture2D * src, Texture2D * dst);

    float       last_time_;

    Matrix4D    prevView_;
    Vector4D    info_;

    Texture2D * grassPack_, * fungus_, * weight_, * grass_, * dirt_;

    glShaderManager SM;

    glShader  * terrainShader_,
              * grassShader_;

    GLint       uniform_weight_, uniform_fungus_,
                uniform_dirt_, uniform_grass_,
                uniform_grassPack_,
                uniform_alphaBooster_,
                uniform_elapsedTime_;

    LPPortalManager  portalsManager_;
    Portal3D  * portals_;

    HeightMap3D heightMapInfo_;

    float   alphaReference_,
            alphaBooster_,
            heightOffset_;

    bool    occlusionCullCells_,
            frustumCullCells_,
            alphaToCoverage_,
            queryClock_,
            drawGrass_,
            drawWireFrame_,
            dirty_;

    static vector <char *> eventmap;
    virtual int getEventID (char * eventname) GET_EVENT_ID(eventname)
};

#endif

