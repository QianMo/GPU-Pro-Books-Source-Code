
#include "SceneGraph.h"

#include <GL/gl.h>

Terrain3D::Terrain3D ()
{
    occlusionCullCells_ = false; //  true;
    frustumCullCells_   = false; // since frustm.update () function is not currently being called
    alphaToCoverage_    =  true;
    alphaReference_     = 0.25f;
    alphaBooster_       =  1.5f;

    heightOffset_       =  0.0f;
    drawWireFrame_      = false;
    drawGrass_          =  true;
    queryClock_         = false;
    dirty_              = false;
    portals_            =  NULL;
    last_time_          =  0.0f;
}

Terrain3D::~Terrain3D ()
{
    SAFEDELETEARRAY (portals_);
    portalsManager_.clear ();

    SAFEDELETE (terrainShader_);
    SAFEDELETE (grassShader_);
}

void Terrain3D::parse (xmlNodePtr pXMLNode)
{
    char * val = NULL;
    int   vali;
    bool  valb;
    float valf;

    val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"occlusionCull");
    if (val)
    {
        parseBoolean (valb, val);
        setOcclusionCullingFlag (valb);

        xmlFree (val);
    }
    val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"frustumCull");
    if (val)
    {
        parseBoolean (valb, val);
        setFrustumCullingFlag (valb);

        xmlFree (val);
    }
    val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"alphaCoverage");
    if (val)
    {
        parseBoolean (valb, val);
        setAlphaToCoverageFlag (valb);

        xmlFree (val);
    }
    val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"alphaReference");
    if (val)
    {
        parseFloat (valf, val);
        setAlphaReference (valf);

        xmlFree (val);
    }
    val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"alphaBooster");
    if (val)
    {
        parseFloat (valf, val);
        setAlphaBooster (valf);

        xmlFree (val);
    }
    val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"heightOffset");
    if (val)
    {
        parseFloat (valf, val);
        heightOffset_ = valf;

        xmlFree (val);
    }
    val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"drawWireFrame");
    if (val)
    {
        parseBoolean (valb, val);
        setWireFrameFlag (valb);

        xmlFree (val);
    }
    val = (char *)xmlGetProp (pXMLNode, (xmlChar *)"drawGrass");
    if (val)
    {
        parseBoolean (valb, val);
        setDrawGrassFlag (valb);

        xmlFree (val);
    }

    Node3D::parse (pXMLNode);
}

void Terrain3D::init (void)
{
    const GLubyte *str = glGetString (GL_EXTENSIONS);
    if (! (strstr ((const char *) str, "GL_ARB_occlusion_query") != NULL))
    {
        EAZD_TRACE ("Terain3D::init() : ERROR - Occlusion Query is not supported");
        exit (EXIT_FAILURE);
    }

    CHECK_GL_ERROR ();

    Texture2D * randommap   = new Texture2D ((char *) "grass/randommap.tga");
    Texture2D * heightmap   = new Texture2D ((char *) "grass/heightmap.jpg");
    Texture2D * coveragemap = new Texture2D ((char *) "grass/coverage.png");
    Texture2D * watermap    = new Texture2D ((char *) "grass/watermap.jpg");

    dirt_ = new Texture2D ((char *) "grass/dirt.dds");
    dirt_->setAnisotropy (2);

    fungus_ = new Texture2D ((char *) "grass/fungus.dds");
    fungus_->setAnisotropy (2);

    grass_ = new Texture2D ((char *) "grass/grasslayer.dds");
    grass_->setAnisotropy (2);

    weight_ = watermap;
    weight_->setTiling (GL_CLAMP, GL_CLAMP);
    weight_->setAnisotropy (2);

    grassPack_ = new Texture2D ((char *) "grass/grassPack.dds");
    grassPack_->setTiling (GL_REPEAT, GL_CLAMP_TO_EDGE);
    grassPack_->setAnisotropy (4);

    CHECK_GL_ERROR ();

    // load the terain shader
    if ((terrainShader_ = SM.loadfromFile ((char *) "grass/terrain.vert", (char *) "grass/terrain.frag")) == 0)
    {
        EAZD_TRACE ("Terrain3D::init() : ERROR - Could not load, compile or link terrain shader");
        exit (EXIT_FAILURE);
    }
    else
    {
        uniform_weight_ = terrainShader_->GetUniformLocation ("weight");
        uniform_fungus_ = terrainShader_->GetUniformLocation ("fungus");
        uniform_dirt_   = terrainShader_->GetUniformLocation ("dirt");
        uniform_grass_  = terrainShader_->GetUniformLocation ("grass");
    }
    CHECK_GL_ERROR ();

    // load the grass shader
    if ((grassShader_ = SM.loadfromFile ((char *) "grass/grass.vert", (char *) "grass/grass.frag")) == 0)
    {
        EAZD_TRACE ("Terrain3D::init() : ERROR - Could not load, compile or link grass shader");
        exit (EXIT_FAILURE);
    }
    else
    {
        uniform_grassPack_    = grassShader_->GetUniformLocation ("grass");
        uniform_alphaBooster_ = grassShader_->GetUniformLocation ("alphaBooster");
        uniform_elapsedTime_  = grassShader_->GetUniformLocation ("elapsedTime");
    }
    CHECK_GL_ERROR ();

    if (! heightMapInfo_.init (heightmap, heightOffset_, watermap))
    {
        EAZD_TRACE ("Terrain3D::init() : ERROR - Could not init heightmap");
        exit (EXIT_FAILURE);
    }
    CHECK_GL_ERROR ();

    portals_ = new Portal3D[CELL_COUNT];
    int     xOf    = 0, yOf = 0,
            xRatio = HEIGHT_MAP_WIDTH / CELL_COLUMN_COUNT,
            yRatio = HEIGHT_MAP_DEPTH / CELL_ROW_COUNT;

    register int index = 0, x = 0, y = 0;

    for (y = 0; y < CELL_ROW_COUNT; y++)
    for (x = (CELL_COLUMN_COUNT - 1); x > -1; x--)
    {
        index = y * CELL_COLUMN_COUNT + x;
        portalsManager_.push_back (portals_ + index);

        xOf = (x == (CELL_COLUMN_COUNT - 1)) ? -1 : 0;
        yOf = (y == (CELL_ROW_COUNT    - 1)) ? -1 : 0;

        if (! portals_[index].init (heightMapInfo_,
                Vector2D (x * xRatio + xOf, y * yRatio + yOf),
                randommap, coveragemap))
            EAZD_TRACE ("Terrain3D::init() : ERROR - Error setting up portal " << index);
    }

    TerrainCell3D::setupIndices (true); // compile);
    if (! appendToGreenChannel (heightmap, watermap))
        EAZD_TRACE ("Terrain3D::init() : ERROR - Error appendToGreenChannel");

    last_time_ = world->getTime () * 1000.f;

    Node3D::init ();
}

void Terrain3D::draw (void)
{
    Frustum3D frustum;

    // calculate the amount of elapsed seconds
    float   elapsedTime = (float) world->getTime () - last_time_ / 1000.f;

    draw (frustum, elapsedTime);

    last_time_ = world->getTime () * 1000.f;
}

void Terrain3D::draw (Frustum3D & frustum, float elapsedTime)
{
    if (portals_)
    {
        GrassCell3D   * gCell    = NULL;
        Portal3D      * portal   = NULL;
        int         index        = 0,
                    x            = 0,
                    y            = 0;

        GLfloat modl[16];
        glGetFloatv (GL_MODELVIEW_MATRIX, modl);
        Matrix4D modlMat = Matrix4D (modl);

        if (prevView_ != modlMat)
        {
            Vector3D viewerPosition = world->getActiveCamera ()->getCOP ();
            prevView_ = modlMat;

            for (y = 0; y < CELL_ROW_COUNT; y++)
            for (x = 0; x < CELL_COLUMN_COUNT; x++)
            {
                index = y * CELL_COLUMN_COUNT + x;
                portals_[index].setVisiblePixelsCount (frustumCullCells_ ? frustum.BBoxInFrustum (portals_[index].getBBox ()) : true);
                portalsManager_[index].set (viewerPosition.distance (portals_[index].getBBox ().getCenter ()), portals_ + index);
            }

// TODO: does not compile
         // sort (portalsManager_.begin (), portalsManager_.end ());

            if (occlusionCullCells_)
                queryClock_ = true;
        }
        else
        {
            if (queryClock_)
            {
                for (x = 3; x < CELL_COUNT; x++)
                    portalsManager_[x].getObject ()->endOcclusionQuery ();
            }
            queryClock_ = false;
        }
    }
    dirty_ = true;

    drawSimple (elapsedTime);
}

void Terrain3D::drawSimple (float elapsedTime)
{
    info_ = Vector4D (0,0,0,0);

    if (portals_)
    {
        GrassCell3D   * gCell    = NULL;
        Portal3D      * portal   = NULL;
        int         visibleCells = 0,
                    index        = 0,
                    x            = 0,
                    y            = 0;

        glPolygonMode (GL_FRONT_AND_BACK, drawWireFrame_ ? GL_LINE : GL_FILL);

        glActiveTextureARB (GL_TEXTURE0_ARB + 0);
        glBindTexture (GL_TEXTURE_2D, weight_->getID ());

        glActiveTextureARB (GL_TEXTURE0_ARB + 1);
        glBindTexture (GL_TEXTURE_2D, fungus_->getID ());

        glActiveTextureARB (GL_TEXTURE0_ARB + 2);
        glBindTexture (GL_TEXTURE_2D, dirt_->getID ());

        glActiveTextureARB (GL_TEXTURE0_ARB + 3);
        glBindTexture (GL_TEXTURE_2D, grass_->getID ());

        terrainShader_->begin ();
        terrainShader_->setUniform1i (0, 0, uniform_weight_);
        terrainShader_->setUniform1i (0, 1, uniform_fungus_);
        terrainShader_->setUniform1i (0, 2, uniform_dirt_);
        terrainShader_->setUniform1i (0, 3, uniform_grass_);

        if (queryClock_ && dirty_)
        {
            for (x = 0; x < CELL_COUNT; x++)
            {
                portalsManager_[x].getObject ()->startOcclusionQuery ();
                visibleCells += (portalsManager_[x].getObject ()->getVisiblePixelsCount () != 0);
            }
        }
        else
            for (x = 0; x < CELL_COUNT; x++)
                visibleCells += portalsManager_[x].getObject ()->draw (Portal3D::TERRAIN) ? 1 : 0;

        terrainShader_->end ();

        if (drawGrass_)
        {
            if (alphaToCoverage_)
            {
                glEnable (GL_MULTISAMPLE_ARB);
                glEnable (GL_SAMPLE_ALPHA_TO_COVERAGE_ARB);
            }

            glEnable (GL_ALPHA_TEST);
            glDisable (GL_CULL_FACE);

            glAlphaFunc (GL_GEQUAL, alphaReference_);

            grassShader_->begin ();
            grassShader_->setUniform1f (0, alphaBooster_, uniform_alphaBooster_);
            grassShader_->setUniform1f (0, elapsedTime,  uniform_elapsedTime_);
            grassShader_->setUniform1i (0, 0, uniform_grassPack_);

            glActiveTextureARB (GL_TEXTURE0_ARB + 0);
            glEnable (GL_TEXTURE_2D);
            glBindTexture (GL_TEXTURE_2D, grassPack_->getID ());

            for (x = 0; x < CELL_COUNT; x++)
            {
                gCell = & portalsManager_[x].getObject ()->getGrassCell ();
                if (portalsManager_[x].getObject ()->draw (Portal3D::GRASS))
                    info_.x += gCell->getTriangleCount ();
            }

            if (alphaToCoverage_)
            {
                glDisable (GL_MULTISAMPLE_ARB);
                glDisable (GL_SAMPLE_ALPHA_TO_COVERAGE_ARB);
            }

            glActiveTextureARB (GL_TEXTURE0_ARB);
            glDisable (GL_TEXTURE_2D);
            glBindTexture (GL_TEXTURE_2D, 0);

            grassShader_->end ();

            glEnable (GL_CULL_FACE);
            glDisable (GL_ALPHA_TEST);
        }

        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);

        if (getDebug ())
        {
            for (y = 0; y < CELL_ROW_COUNT; y++)
            for (x = 0; x < CELL_COLUMN_COUNT; x++)
                portals_[y * CELL_COLUMN_COUNT + x].draw (Portal3D::BBOX);

            heightMapInfo_.draw ();
        }

        info_.x += (TILE_COLUMN_COUNT - 1) * TILE_ROW_COUNT * visibleCells;
        info_.y  = visibleCells * 100 / CELL_COUNT;
    }

    dirty_ = false;
}

bool  Terrain3D::appendToGreenChannel (Texture2D * src, Texture2D * dst)
{
    GLubyte   * hmBytes = (GLubyte *) dst->getDataPtr (),
              * wmBytes = (GLubyte *) src->getDataPtr ();

    int     hmByteCount = dst->getNumComponents (),
            wmByteCount = src->getNumComponents (),
            width       = dst->getWidth (),
            height      = dst->getHeight (),
            index       = 0;

    if (! wmBytes || ! hmBytes)
    {
        EAZD_TRACE ("Terrain3D::appendToGreenChannel() : ERROR - Null src or dst content");
        return false;
    }

    if (! height == src->getHeight () || ! width == src->getWidth ())
    {
        EAZD_TRACE ("Terrain3D::appendToGreenChannel() : ERROR - dst has unexpected width or height");
        return false;
    }

    for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
    {
        index = y * width + x;
        hmBytes[index * hmByteCount + 1] = wmBytes[index * wmByteCount];
    }

    return true;
}

void Terrain3D::adjustCameraYLocation (Camera3D & camera, float cameraRadius)
{
#if 0
    Vector3D position   = camera.getCOP ();
    float   difference  = heightMapInfo_getInterpolatedHeight (position) -
                            position.y + cameraRadius;

    if (difference > 0.1f)
        camera.elevate (difference);
#endif
}

void  Terrain3D::resetRenderingRoutine ()
{
    prevView_.identity ();
    queryClock_ = false;
}

