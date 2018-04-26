/***************************************************************************/
/* sceneLoader.cpp                                                         */
/* -----------------------                                                 */
/*                                                                         */
/* This is the main entry point into my current OpenGL framework (entitled */
/*    simply enough "sceneLoader").  This code parses command-line params, */
/*    initializes the scene, initializes OpenGL, checks for all hardware   */
/*    functionality & OpenGL driver support needed for the demo, and fires */
/*    of the main GLUT window.  It also contain a basic GLUT display       */
/*    callback, which checks to see which render mode we're currently set  */
/*    to use.  Except for a simplistic diffuse rendering of the scene, all */
/*    code for various rendering modes are in separate files inside the    */
/*    directory "RenderingTechniques/"                                     */
/*                                                                         */
/* Chris Wyman (02/01/2008)                                                */
/***************************************************************************/

#include "stencilMultiResSplatting.h"
#include "RenderingTechniques/renderingData.h"
#include "RenderingTechniques/renderingMethods.h"

Scene *scene = 0;
FrameRate *frameSpeed = 0;
FrameRate *frameSpeedSlow = 0;
FrameGrab *frameGrab  = 0;
extern RenderingData *data;

UIInt *displayID=0;
int lastDisplay=-1;

void Display_WithReflectiveShadowMap( void );
void Display_WithReflectiveShadowMapGathering( void );
void Display_WithMultiResGather_NoInterpolation( void );
void Display_WithMultiResGather_StencilRefinement( void );

char *GetWindowName( int identifier )
{
    #define NUM_TITLES   5 
	static char buf[NUM_TITLES][128] = {
		"(1) Basic, Diffuse Scene Rendering",
		"(2) Reflective Shadow Maps (using full-screen splatting)",
		"(3) Reflective Shadow Maps (using full-screen gathering)",
		"(4) Reflective Shadow Maps (using multiresolution splatting & NO INTERPOLATION during upsampling)",
		"(5) Reflective Shadow Maps (using multiresolution splatting)"
	};
	static char newTitle[256];
	sprintf( newTitle, "Error!  Unknown title text!!!  (Id: %d)", identifier );
	return identifier < 0 || identifier >= NUM_TITLES ? newTitle : buf[identifier];
	#undef NUM_TITLES
}


void DisplayCallback( void )
{
	// Make sure the title of the window stays up-to-date if we change render modes
	if (*displayID != lastDisplay)
	{
		glutSetWindowTitle( GetWindowName( *displayID ) );
		lastDisplay = *displayID;
	}

	// Start timing this frame
	frameSpeed->StartFrame();
	frameSpeedSlow->StartFrame();

	// Draw the scene using the appropriate render mode
	if (*displayID > 0)
	{
		switch( *displayID )
		{
		case 1:
			Display_WithReflectiveShadowMap();
			break;
		case 2:
			Display_WithReflectiveShadowMapGathering();
			break;
		case 3:
			Display_WithMultiResGather_NoInterpolation();
			break;
		case 4:
			Display_WithMultiResGather_StencilRefinement();
			break;
		}
	}
	else  // This is the render mode for a simple, diffuse scene
	{
		// Create shadow map(s) 
		for (int i=0; i < scene->NumLightsEnabled(); i++)
			scene->CreateShadowMap( data->fbo->shadowMap[i], 
									data->param->shadowMatrixTrans[i], 
									i, // Light Number
									*(data->param->shadowMapBias) );

		// Draw the scene with shadowing enabled
		data->fbo->mainWin->BindBuffer();
		data->fbo->mainWin->ClearBuffers();
		glLoadIdentity();
		scene->LookAtMatrix();
		scene->SetupEnabledLightsWithCurrentModelview();
		scene->Draw( MATL_FLAGS_USESHADOWMAP );

		data->fbo->mainWin->UnbindBuffer();
	}

	// Copy the final rendering to the screen
	data->fbo->mainWin->DisplayAsFullScreenTexture( FBO_COLOR0, false );

	// If a screen capture is desired, do it now (before writing timer text)
	if (data->ui->captureScreen)
	{
		frameGrab->CaptureFrame();
		data->ui->captureScreen = false;
	}

	// If the user wants the help screen then display it.
	if (data->ui->displayHelp)
	{
		//printf(" needs updates? %d\n", data->ui->updateHelpScreen ? 1 : 0 );
		if (data->ui->updateHelpScreen)
			SetupHelpScreen( data->fbo->helpScreen );
		DisplayHelpScreen( data->fbo->helpScreen );
	}

	// Output the timer text
	float slowSpeed = frameSpeedSlow->EndFrame();
	if (slowSpeed < 10)
	{
		DisplayTimer( slowSpeed, 1024, 1024 );
		frameSpeed->EndFrame();
	}
	else
		DisplayTimer( frameSpeed->EndFrame(), 1024, 1024 );
	glFlush();
	glutSwapBuffers();
}

// Check to make sure whoever runs the demo has a capable card...
void CheckMachineCapabilities( void )
{
	printf("(+) Checking for necessary hardware requirements.....");

	if ( GLEE_VERSION_2_0 && GLEE_EXT_gpu_shader4 && 
		 (!GLEE_EXT_geometry_shader4 || !GLEE_EXT_texture_array || !GLEE_NV_transform_feedback) )
	{
		printf("\n    (-) NOTE: This demo was developed on NVIDIA cards that supported\n");
		printf("              EXT_geometry_shader4, EXT_texture_array, and \n");
		printf("              NV_transform_feedback.  Your graphics card/driver does not\n");
		printf("              support one or more of these extensions.  While the demo\n");
		printf("              *should not* require these extensions, I have not tested the\n");
		printf("              code to make sure all dependancies have been removed.\n");
		return;
	}

	if ( GLEE_VERSION_2_0 && GLEE_EXT_gpu_shader4 )
	{
		printf("  OK.\n");
		return;
	}
	else printf("  MISSING!\n");

	printf("    (-) Running: OpenGL v%s, GLEE v5.4, GLSL v%s \n", 
		glGetString( GL_VERSION ),
		glGetString(GL_SHADING_LANGUAGE_VERSION) );
	printf("    (-) Compiled against: GLUT v3.7.6 \n");
	printf("    (-) Running on a: %s %s\n", glGetString( GL_VENDOR ), glGetString( GL_RENDERER ));
	glGetError();
	
	/* check if the machine is equipped to run the demo */
	if (!GLEE_VERSION_2_0)
		printf("    (-) FATAL ERROR: Demo requires OpenGL 2.0 or higher!\n");
	if (!GLEE_EXT_gpu_shader4)
		printf("    (-) FATAL ERROR: Demo requires OpenGL Extension EXT_gpu_shader4!\n");

	// Make sure we don't quickly close the program without letting
	//    the user read about their horribly deficient hardware!
	printf("(+) Press enter to exit...\n" );
	getchar();

	// OK.  Now quit.
	exit(-1);
}


// The main entry point.
int main(int argc, char* argv[] )
{
	bool verbose = false;
	char scenefile[ 256 ];
	printf("***************************************************************************\n");
	printf("* Fast, Stencil-Based Multiresolution Splatting for Indirect Illumination *\n");
	printf("*                        Demo for GPU Pro Article                         *\n");
	printf("*                Chris Wyman, Greg Nichols and Jeremy Shopf               *\n");
	printf("*                           University of Iowa                            *\n");
	printf("*      More Information: http://www.cs.uiowa.edu/~cwyman/pubs.html        *\n");
	printf("***************************************************************************\n");

	// Check to make sure the user gave a scene file.  If not, prompt for one.
	printf("(+) Parsing any command line parameters...\n");
	if (argc < 2)
	{
		printf("\nUsage: multiResSplatting.exe <sceneFile>\n\n", argv[0]);
		printf("    (-) No scene file specified!  Using default \"cbox_meters.txt\"?\n");
		sprintf( scenefile, "cbox_meters.txt" );
	}
	else  // Grab the scene from the command line.
	{
		for( int i=1; i < argc; i++ )
		{
			if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help") || 
				!strcmp(argv[i], "-?") || !strcmp(argv[i], "/?") )
			{
				printf("Usage: %s <sceneFile>\n", argv[0]);
				exit(0);
			}
			else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "-verbose"))
				verbose = true;
			else
			{
				strncpy( scenefile, argv[i], 255 );
				break;
			}
		}
	}

	// Load the scene from the file.  Unfortunately, this needs to come prior
	//   to initializing the GL window and checking for an appropriate hardware
	//   configuration.  It should really be the other way around for a demo...
	scene = new Scene( scenefile, verbose );

	printf("(+) Initializing OpenGL state...\n");
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA );
	glutInitWindowSize( scene->GetWidth(), scene->GetHeight() );
	glutCreateWindow( "DEMO: Multiresolution Splatting for Indirect Illumination" );
	GLeeInit();

	// Check hardware compatibility... before we crash from NULL GLEW pointers.
	CheckMachineCapabilities();

	// Set the GLUT callback functions 
	glutDisplayFunc( DisplayCallback );
	glutReshapeFunc( ReshapeCallback );
	glutIdleFunc( IdleCallback );
	glutMouseFunc( MouseButtonCallback );
	glutMotionFunc( MouseMotionCallback );
	glutKeyboardFunc( KeyboardCallback );
	glutSpecialFunc( SpecialKeyboardCallback ); 

	// Other program setup 
	frameSpeed = new FrameRate( 30 );
	frameSpeedSlow = new FrameRate( 5 );
	frameGrab  = new FrameGrab();

	// Make sure any preprocessing that needs to be done occurs
	scene->Preprocess();

	// Initialized data needed by the various rendering modes
	InitializeRenderingData();

	// Get a pointer to the Display ID
	displayID = (UIInt *)scene->GetSceneVariable( "displayid", new UIInt( 0 ) );

	// Set the type of movement interaction (so it can be displayed correctly on the helpscreen)
	data->ui->movementType = 1;
	if (!strcmp( scenefile, "cbox_meters.txt" )) data->ui->movementType = 2;
	if (!strcmp( scenefile, "buddhaCaustics.txt" )) data->ui->movementType = 2;

	// Allow colors outside the normal range [0..1]
	glClampColorARB( GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE );
    glClampColorARB( GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE );
    glClampColorARB( GL_CLAMP_READ_COLOR_ARB, GL_FALSE );

	// Run the display!
	glEnable( GL_DEPTH_TEST );
	glEnable( GL_LIGHTING );
	glEnable( GL_NORMALIZE );
	glClearStencil( 0x0 );
	glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
	glBlendFunc( GL_ONE, GL_ONE );
	if (scene->HasCulling()) glEnable( GL_CULL_FACE );

	// Enable all the lights needed
	for (int i=0; i < *data->ui->numLightsUsed; i++)
	{
		GLLight *lght = scene->GetLight(i);
		if (lght) lght->Enable();
	}

	// This is a bit of a nonsense call, we don't really need
	//   most of what this function does.  However, it DOES
	//   update the state of the internal scene variable "numLightsEnabled"
	//   which, somehow, appears out-of-date on some non-debug 
	//   (i.e., Release) builds, causing the program to crash
	//   immediately upon entering the glutMainLoop().
	// I really need to just fix this by changing the abstraction
	//   for how lights are enabled above (and elsewhere).  But
	//   when the bug shows up just as the demo is almost out 
	//   the door...
	scene->SetupEnabledLightsWithCurrentModelview();

	// Start rendering!
	glutMainLoop();

	// Should never be hit, but compilers seem to need it...
	return 0;
}
