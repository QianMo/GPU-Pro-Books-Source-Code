/***************************************************************************/
/* initializeRenderingData.cpp                                             */
/* -----------------------                                                 */
/*                                                                         */
/* Initialization routines for data needed for all the rendering methods   */
/*    in this directory.                                                   */
/*                                                                         */
/* Chris Wyman (02/23/2008)                                                */
/***************************************************************************/


#include "stencilMultiResSplatting.h"
#include "renderingData.h"
#include "renderingMethods.h"

extern Scene *scene;
RenderingData *data = 0;


// This initializes the RenderingData structure declared above, which is my way
//    of using global data without a lot of the problems that typically
//    are associated with it (i.e., lists of variables at the top of each
//    file, poor organization, etc., etc.).
// For data or data structures not directly associated with the particular
//    scene we have loaded, this is where they are stored.
// In particular, below are the framebuffer objects, vertex buffer objects,
//    non-scene specific textures, shaders and parameters specific to
//    our experimental shaders, and program-specific UI data.
void InitializeRenderingData( void )
{
	// Check to make sure we only initialize once.
	if (data) return;


	/////////////////////////////////////////////////////////////////////////////////////////
	//
	// Allocate global data structures
    //
	/////////////////////////////////////////////////////////////////////////////////////////

	data = new RenderingData();
	data->fbo    = new FrameBufferData();
	data->param  = new ParameterData();
	data->shader = new ShaderData();
	data->tex    = new TextureData();
	data->vbo    = new VertexBufferData();
	data->glID   = new OtherGLIDData();
	data->ui     = new UIData();
	data->rng	 = new Random();

	/////////////////////////////////////////////////////////////////////////////////////////
	//
	// Setup VPL counts;
    //
	/////////////////////////////////////////////////////////////////////////////////////////

	data->param->maxVplCount  = 4096;
	data->param->vplCountSqrt = 16;
	data->param->vplCount     = data->param->vplCountSqrt*data->param->vplCountSqrt;
	data->param->vplDelta     = 1.0/data->param->vplCountSqrt;
	data->param->vplOffset    = 0.5/data->param->vplCountSqrt;


	/////////////////////////////////////////////////////////////////////////////////////////
	//
	// Declare our framebuffer objects that various rendering modes will draw to
    //
	/////////////////////////////////////////////////////////////////////////////////////////

	// Framebuffer to render as the main window (later downsampled for antialiasing).
	//    Probably we don't need this, but I've found it convenient to use have the
	//    "main window" be a FBO for a large number of reasons.  However, it does
	//    definitely slow the the program down....  For speed, draw to the real window.
	data->fbo->mainWin = new FrameBuffer( GL_TEXTURE_2D, scene->GetWidth(), scene->GetHeight(), -1, 
										  GL_RGBA16F_ARB, 1, 1, false, "Main Render Window" );
	data->fbo->mainWin->CheckFramebufferStatus( 1 );

	// Framebuffer to render help instructions.  This probably isn't needed either,
	//    but in my experience GLUT draws text exceedingly slowly, and I'm trading
	//    space (for this FBO) to reduce per-frame text rendering costs when displaying
	//    the help screen.  For speed, use a really good UI/text rendering package.
	data->fbo->helpScreen = new FrameBuffer( GL_TEXTURE_2D, scene->GetWidth(), scene->GetHeight(), -1, 
										  GL_RGBA, 1, 0, false, "Help Screen" );
	data->fbo->helpScreen->CheckFramebufferStatus( 1 );

	// Framebuffer to store the shadow map(s).  These are pretty big, perhaps bigger
	//    than they need to be, but I see no reason to have poor quality shadows
	//    distracting from the evaluation of our caustics.
	if (scene->GetNumLights() > 1)
		printf("    (-) Allocating buffers for %d light sources...\n", scene->GetNumLights());
	for (int i=0; i < scene->GetNumLights(); i++)
	{
		data->fbo->shadowMap[i] = new FrameBuffer( GL_TEXTURE_2D, 2*scene->GetWidth(), 2*scene->GetHeight(), -1, 
												GL_ALPHA, 1, 1, false, "Shadow Map" );
		data->fbo->shadowMap[i]->CheckFramebufferStatus( 1 );
	}

	// Framebuffer that compactly stores our selected VPLs, so that we can
	//   efficiently access them in a texture cache-friendly manner.
	data->fbo->compactVPLTex = new FrameBuffer( GL_TEXTURE_2D, data->param->maxVplCount, 3, -1, 
	//data->fbo->compactVPLTex = new FrameBuffer( GL_TEXTURE_2D, data->param->vplCount, 3, -1, 
		                                        GL_RGBA16F_ARB, 1, 0, false, "Compact VPL Texture" );
	data->fbo->compactVPLTex->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	data->fbo->compactVPLTex->SetAttachmentClamping( FBO_COLOR0, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE );
	data->fbo->compactVPLTex->CheckFramebufferStatus( 1 );

	// Framebuffer that stores our reflective shadow map.  This actually contains three
	//   render targets.  One for fragment position, one for fragment normal, and one for
	//   reflected flux.
	data->fbo->reflectiveShadowMap = new FrameBuffer( GL_TEXTURE_2D, scene->GetWidth(), scene->GetHeight(), -1, 
												GL_RGBA16F_ARB, 3, 1, true, "Reflective Shadow Map" );
	data->fbo->reflectiveShadowMap->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	data->fbo->reflectiveShadowMap->SetAttachmentFiltering( FBO_COLOR1, GL_NEAREST, GL_NEAREST );
	data->fbo->reflectiveShadowMap->SetAttachmentFiltering( FBO_COLOR2, GL_NEAREST, GL_NEAREST );
	data->fbo->reflectiveShadowMap->CheckFramebufferStatus( 1 );

	// Framebuffer that stores our depth-derivative mipmap.  This is recomputed every frame
	//    and is used to determine when to refine our multiresolution splats
	data->fbo->depthDeriv = new FrameBuffer( GL_TEXTURE_2D, scene->GetWidth(), scene->GetHeight(), -1, 
												GL_LUMINANCE16F_ARB, 1, 0, true, "Depth Derivative MipMap" );
	data->fbo->depthDeriv->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	data->fbo->depthDeriv->SetAttachmentClamping ( FBO_COLOR0, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE );
	data->fbo->depthDeriv->CheckFramebufferStatus( 1 );

	// Framebuffer that stores our surface normal mipmap.  This is recomputed every frame
	//    and is used to determine when to refine our multiresolution splats
	data->fbo->normalThreshMipMap = new FrameBuffer( GL_TEXTURE_2D, scene->GetWidth(), scene->GetHeight(), -1, 
												GL_RGBA16F_ARB, 1, 0, true, "Normal Threshold MipMap" );
	data->fbo->normalThreshMipMap->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	data->fbo->normalThreshMipMap->SetAttachmentClamping ( FBO_COLOR0, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE );
	data->fbo->normalThreshMipMap->CheckFramebufferStatus( 1 );

	// This is our multiresolution render buffer that stores indirect illumination.
	//    It also has a stencil plane to allow early z-culling for efficiency when
	//    splatting.
	data->fbo->multiResIllumBuffer = new FrameBuffer( GL_TEXTURE_2D, 2*scene->GetWidth(), scene->GetHeight(), -1, 
												GL_RGBA16F_ARB, 1, 2, false, "Subsplat Accumulation" );
	data->fbo->multiResIllumBuffer->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	data->fbo->multiResIllumBuffer->SetAttachmentClamping( FBO_COLOR0, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE );
	data->fbo->multiResIllumBuffer->CheckFramebufferStatus( 1 );

	// When upsampling our multires buffer for final display, we need to do some
	//    ping-ponging to access the appropriate data.  This buffer is identically
	//    sized with our multres render buffer.
	data->fbo->pingPongBuffer = new FrameBuffer( GL_TEXTURE_2D, 2*scene->GetWidth(), scene->GetHeight(), -1, 
												GL_RGBA16F_ARB, 1, 0, false, "Subsplat Ping Pong Buffer" );
	data->fbo->pingPongBuffer->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	data->fbo->pingPongBuffer->SetAttachmentClamping( FBO_COLOR0, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE );
	data->fbo->pingPongBuffer->CheckFramebufferStatus( 1 );

	// Framebuffer that stores our G-buffer for deferred rendering from the eye.  This 
	//   actually contains three render targets.  One for fragment position, one for 
	//   fragment normal, and one for surface albedo.
	data->fbo->deferredGeometry = new FrameBuffer( GL_TEXTURE_2D, scene->GetWidth(), scene->GetHeight(), -1, 
												GL_RGBA16F_ARB, 3, 1, false, "G-Buffer (Deferred Shading Buffer)" );
	data->fbo->deferredGeometry->SetAttachmentClamping( FBO_COLOR0, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE );
	data->fbo->deferredGeometry->SetAttachmentClamping( FBO_COLOR1, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE );
	data->fbo->deferredGeometry->SetAttachmentClamping( FBO_COLOR2, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE );
	data->fbo->deferredGeometry->CheckFramebufferStatus( 1 );


	/////////////////////////////////////////////////////////////////////////////////////////
	//
	// Initialize assorted static textures here.  In particular, setup a "spotlight" texture
	//    and a University of Iowa logo texture for use on the help-screen.
    //
	/////////////////////////////////////////////////////////////////////////////////////////

	data->tex->spotlight = scene->GetNamedTexture( "spotlight" );
	if (!data->tex->spotlight)
	{
		char *file = scene->paths->GetTexturePath( "spot_white.ppm" );
		data->tex->spotlight = new GLTexture( file, TEXTURE_REPEAT_S | TEXTURE_REPEAT_T | TEXTURE_MIN_LINEAR_MIP_LINEAR );
		free( file );
	}

	// Load the Iowa logo (used on the help screen)
	{
		char *file = scene->paths->GetTexturePath( "iowaLogo.ppm" );
		data->tex->iowaLogo = new GLTexture( file, TEXTURE_REPEAT_S | TEXTURE_REPEAT_T | TEXTURE_MIN_LINEAR_MIP_LINEAR );
		free( file );
	}

	// Save a simple glOrtho projection matrix, which we use in a couple of our shaders.
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0,1,0,1);
	glGetFloatv( GL_MODELVIEW_MATRIX, data->param->gluOrtho );
	glPopMatrix();

	// Initialize starting UI information
	data->ui->updateHelpScreen = true;
	data->ui->displayHelp = true;
	data->ui->captureScreen = false;


	/////////////////////////////////////////////////////////////////////////////////////////
	//
	// Initialize "scene variables," those that are declared and modifiable in the scene
	//    file.  Also, if the scene file neglects to define them we give them a default here
    //
	/////////////////////////////////////////////////////////////////////////////////////////

	data->ui->benchmark             = (UIBool *)scene->GetSceneVariable( "benchmark",              new UIBool( false ) );
	data->ui->animation             = (UIBool *)scene->GetSceneVariable( "animation",              new UIBool( true ) );
	data->param->shadowMapBias      = (UIFloat *)scene->GetSceneVariable( "shadowmapbias",         new UIFloat( -0.005 ) );
	data->param->lightFOV           = (UIFloat *)scene->GetSceneVariable( "lightfov",              new UIFloat( 90.0 ) );
	data->ui->translationSpeed      = (UIFloat *)scene->GetSceneVariable( "translationSpeed",      new UIFloat( 0.1 ) );
	data->ui->depthThreshold        = (UIFloat *)scene->GetSceneVariable( "depththreshold",        new UIFloat( 0.1 ) );
	data->ui->normThreshold         = (UIFloat *)scene->GetSceneVariable( "normthreshold",         new UIFloat( 0.1 ) );
	data->ui->vplIntensityThreshold = (UIFloat *)scene->GetSceneVariable( "vplintensitythreshold", new UIFloat( 1.0 ) );
	data->ui->numLightsUsed         = (UIInt *)scene->GetSceneVariable( "numlightsused",           new UIInt( 1 ) );
	data->ui->uiVPLCount            = (UIInt *)scene->GetSceneVariable( "vplcount",                new UIInt( data->param->vplCountSqrt ) );


	/////////////////////////////////////////////////////////////////////////////////////////
	//
	// Allocate, load, and initialize all the shaders needed by our various rendering modes.
    //
	/////////////////////////////////////////////////////////////////////////////////////////

	// A shader that upsamples our multiresolution buffer using no (i.e, nearest neighbor) interpolation
	scene->AddShader( data->shader->upsampleNoInterpolation =
		              new GLSLProgram( "multiresUpsample_noInterp.vert.glsl",
					                   NULL,
									   "multiresUpsample_noInterp.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->upsampleNoInterpolation->SetupAutomatic4x4MatrixBinding( "gluOrtho",  data->param->gluOrtho );
	data->shader->upsampleNoInterpolation->SetProgramDisables( GLSL_BLEND | GLSL_DEPTH_TEST | GLSL_STENCIL_TEST );


	// A shader that upsamples our multresolution buffer using the method proposed by our I3D paper.
	scene->AddShader( data->shader->upsampleI3DInterpolation =
                      new GLSLProgram( "multiresUpsample_i3dInterp.vert.glsl",
					                   NULL,
									   "multiresUpsample_i3dInterp.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->upsampleI3DInterpolation->SetupAutomatic4x4MatrixBinding( "gluOrtho",  data->param->gluOrtho );
	data->shader->upsampleI3DInterpolation->SetProgramDisables( GLSL_BLEND | GLSL_DEPTH_TEST | GLSL_STENCIL_TEST );


	// A shader that blends the results of our upsampled indirect illumination in with the direct illumination
	scene->AddShader( data->shader->modulateLevel0 =
		              new GLSLProgram( "modulateMultResLevel0.vert.glsl",
					                   NULL,
									   "modulateMultResLevel0.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->modulateLevel0->SetupAutomatic4x4MatrixBinding( "gluOrtho",  data->param->gluOrtho );
	data->shader->modulateLevel0->SetProgramEnables( GLSL_BLEND );
	data->shader->modulateLevel0->SetProgramDisables( GLSL_DEPTH_TEST | GLSL_STENCIL_TEST );


	// This shader perfoms deferred shading on a scene that is stored in our G-buffer 
	//    (i.e., data->fbo->deferredGeometry)
	scene->AddShader( data->shader->deferredShade = 
		              new GLSLProgram( "deferredLighting.vert.glsl",
					                   NULL,
									   "deferredLighting.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->deferredShade->SetupAutomatic4x4MatrixBinding( "gluOrtho",  data->param->gluOrtho );
	data->shader->deferredShade->SetupAutomaticBinding( "sceneAmbient", 4, scene->GetSceneAmbient() );
	data->shader->deferredShade->SetupAutomaticTextureBinding( "fragPosition", data->fbo->deferredGeometry->GetColorTextureID( 2 ), GL_TEXTURE0 );
	data->shader->deferredShade->SetupAutomaticTextureBinding( "fragNormal", data->fbo->deferredGeometry->GetColorTextureID( 1 ), GL_TEXTURE1 );
	data->shader->deferredShade->SetupAutomaticTextureBinding( "fragColor", data->fbo->deferredGeometry->GetColorTextureID( 0 ), GL_TEXTURE2 );
	data->shader->deferredShade->SetupAutomaticTextureBinding( "spotLight", data->tex->spotlight->TextureID(), GL_TEXTURE3 );
	data->shader->deferredShade->SetProgramDisables( GLSL_BLEND | GLSL_DEPTH_TEST );


	// This shader creates a compact list (i.e., texture) from the RSM that contains
	//    only data for the selected VPLs.
	scene->AddShader( data->shader->vplCompact = 
		              new GLSLProgram( "vplCompact.vert.glsl",
					                   NULL,
									   "vplCompact.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->vplCompact->SetupAutomatic4x4MatrixBinding( "gluOrtho",  data->param->gluOrtho );
	data->shader->vplCompact->SetupAutomaticTextureBinding( "lightPosition", data->fbo->reflectiveShadowMap->GetColorTextureID( 2 ), GL_TEXTURE1 );
	data->shader->vplCompact->SetupAutomaticTextureBinding( "lightNormal", data->fbo->reflectiveShadowMap->GetColorTextureID( 1 ), GL_TEXTURE2 );
	data->shader->vplCompact->SetupAutomaticTextureBinding( "lightColor", data->fbo->reflectiveShadowMap->GetColorTextureID( 0 ), GL_TEXTURE3 );
	data->shader->vplCompact->SetupAutomaticTextureBinding( "spotTex", data->tex->spotlight->TextureID(), GL_TEXTURE0 );
	data->shader->vplCompact->LinkProgram();


	// This shader gathers indirect illumination into our multiresolution buffer,
	//     but only at locations marked in the associated stencil buffer
	scene->AddShader( data->shader->perSubsplatStenciledGather = 
		              new GLSLProgram( "subsplatStencilGather.vert.glsl",
					                   NULL,
									   "subsplatStencilGather.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->perSubsplatStenciledGather->SetupAutomatic4x4MatrixBinding( "gluOrtho",  data->param->gluOrtho );
	data->shader->perSubsplatStenciledGather->SetupAutomaticTextureBinding( "fragPosition", data->fbo->deferredGeometry->GetColorTextureID( 2 ), GL_TEXTURE0 );
	data->shader->perSubsplatStenciledGather->SetupAutomaticTextureBinding( "fragNormal", data->fbo->deferredGeometry->GetColorTextureID( 1 ), GL_TEXTURE1 );
	data->shader->perSubsplatStenciledGather->SetupAutomaticTextureBinding( "fragColor", data->fbo->deferredGeometry->GetColorTextureID( 0 ), GL_TEXTURE2 );
	data->shader->perSubsplatStenciledGather->SetupAutomaticTextureBinding( "vplCompact", data->fbo->compactVPLTex->GetColorTextureID(), GL_TEXTURE3 );
	data->shader->perSubsplatStenciledGather->SetProgramDisables( GLSL_DEPTH_TEST );
	data->shader->perSubsplatStenciledGather->SetProgramEnables( GLSL_BLEND | GLSL_STENCIL_TEST );

	// This shader is used to create our mipmap that stores depth values.  This
	//    mipmap is used during splat refinement to determine where coarser
	//    samples are sufficient for gathering illumination
	scene->AddShader( data->shader->depthDerivMax = 
		              new GLSLProgram( NULL,
					                   NULL,
									   "depthDerivMaxMipmap.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->depthDerivMax->SetProgramDisables( GLSL_BLEND );

	// This shader is used to create our mipmap that stores normal values.  This
	//    mipmap is used during splat refinement to determine where coarser
	//    samples are sufficient for gathering illumination
	scene->AddShader( data->shader->normalThresholdMipmap = 
		              new GLSLProgram( NULL,
					                   NULL,
									   "normalThresholdMipmap.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->normalThresholdMipmap->SetProgramDisables( GLSL_BLEND );

	// This shader takes as input a linear depth map of the scene and outputs a
	//    texture containing depth derivatives.  The result is used to create our
	//    depth mipmap (which is used during refinement)
	scene->AddShader( data->shader->createRSMDerivativeMap = 
		              new GLSLProgram( NULL,
					                   NULL,
									   "createRSMDerivativeMap.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );

	// This shader takes as input the view's G-buffer (i.e., deferred geometry) 
	//    and creates a normal buffer that can be mipmapped.
	scene->AddShader( data->shader->createNormalThresholdBuffer = 
		              new GLSLProgram( NULL,
					                   NULL,
									   "normalThresholdBuffer.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );

	// This shader is used as a comparison.  Effectively, it performs our most
	//    efficient implementation of "ground truth" reflective shadow mapping.  
	//    In this case, a single full-screen "splat" is drawn and each texel
	//    gathers illumination from all of the selected VPLs.
	scene->AddShader( data->shader->indirectGather = 
		              new GLSLProgram( "vplGather.vert.glsl",
					                   NULL,
									   "vplGather.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->indirectGather->SetupAutomatic4x4MatrixBinding( "gluOrtho",  data->param->gluOrtho );
	data->shader->indirectGather->SetupAutomaticTextureBinding( "fragPosition", data->fbo->deferredGeometry->GetColorTextureID( 2 ), GL_TEXTURE0 );
	data->shader->indirectGather->SetupAutomaticTextureBinding( "fragNormal", data->fbo->deferredGeometry->GetColorTextureID( 1 ), GL_TEXTURE1 );
	data->shader->indirectGather->SetupAutomaticTextureBinding( "fragColor", data->fbo->deferredGeometry->GetColorTextureID( 0 ), GL_TEXTURE2 );
	data->shader->indirectGather->SetupAutomaticTextureBinding( "vplCompact", data->fbo->compactVPLTex->GetColorTextureID(), GL_TEXTURE3 );
	data->shader->indirectGather->SetProgramDisables( GLSL_DEPTH_TEST );
	data->shader->indirectGather->SetProgramEnables( GLSL_BLEND );

	// This shader is used as a comparison.  Effectively, it performs a typical
	//    implementation of "ground truth" reflective shadow mapping.  In this case, 
	//    each VPL spawns a full-screen "splat" that accumulates that VPL's contribution
	//    to the indirect illumination.  This is (quite) slow.
	scene->AddShader( data->shader->indirectAccumulate = 
		              new GLSLProgram( "vplAccumulate.vert.glsl",
					                   NULL,
									   "vplAccumulate.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->indirectAccumulate->SetupAutomatic4x4MatrixBinding( "gluOrtho",  data->param->gluOrtho );
	data->shader->indirectAccumulate->SetupAutomaticTextureBinding( "fragPosition", data->fbo->deferredGeometry->GetColorTextureID( 2 ), GL_TEXTURE7 );
	data->shader->indirectAccumulate->SetupAutomaticTextureBinding( "fragNormal", data->fbo->deferredGeometry->GetColorTextureID( 1 ), GL_TEXTURE1 );
	data->shader->indirectAccumulate->SetupAutomaticTextureBinding( "fragColor", data->fbo->deferredGeometry->GetColorTextureID( 0 ), GL_TEXTURE2 );
	data->shader->indirectAccumulate->SetupAutomaticTextureBinding( "vplCompact", data->fbo->compactVPLTex->GetColorTextureID(), GL_TEXTURE0 );
	data->shader->indirectAccumulate->SetProgramDisables( GLSL_DEPTH_TEST );
	data->shader->indirectAccumulate->SetProgramEnables( GLSL_BLEND );

	// This shader does our "refinement".
	//    In our I3D paper, this required an iterative geometry shader pass.  However,
	//    later work shows that a single-pass stencil-based approach can "refine" our
	//    multiresolution splats in a *significantly* more efficient manner.
	scene->AddShader( data->shader->depthAndNormalStencilRefinement = 
		              new GLSLProgram( NULL,
					                   NULL,
									   "depthAndNormalStencilRefinement.frag.glsl",
									   true, scene->paths->GetShaderPathList() ) );
	data->shader->depthAndNormalStencilRefinement->LinkProgram();


	/////////////////////////////////////////////////////////////////////////////////////////
	//
	// Initialize various other OpenGL identifier data
    //
	/////////////////////////////////////////////////////////////////////////////////////////

	glGenQueries( 1, &data->glID->primCountQuery  );
	glGenQueries( 10, data->glID->timerQuery  );


	/////////////////////////////////////////////////////////////////////////////////////////
	//
	// Done with initialization
    //
	/////////////////////////////////////////////////////////////////////////////////////////

	if (scene->IsVerbose())
		printf("    (-) Finished InitializeRenderingData()...\n");
}



// Nominally, this should free all the memory allocated above.  However, the real reason
//    this is here is that on some graphics cards and drivers (i.e., *mine*) the program
//    freezes on exit unless all memory associated with VBOs is explicitly freed.  
// So the real key is to explicitly free all VBO memory.  If this paticular demo does not
//    use VBOs, this might look like a tremendously stupid function (as I will likely 
//    explicitly free only a tiny subset of my data)
void FreeMemoryAndQuit( void )
{
	// Print a status message
	printf("(+) Exiting program....\n");
	printf("    (-) Freeing resources...\n");

	// Free FBOs
	delete data->fbo->mainWin;
	for (int i=0; i < scene->GetNumLights(); i++)
	{
		delete data->fbo->shadowMap[i];
	}

	// Free the scene (which frees assorted associated data)
	delete scene;

	// Close the GLUT window.
	glutDestroyWindow( glutGetWindow() );

	// Done cleaning up!  Quit!
	printf("    (-) Done.\n");
	exit(0);
}

