
#ifdef WIN32
	#include <windows.h>
#endif

#include <GL/glew.h>
#include <GL/wglew.h>

#include "cglib.h"
#include "DeferredRenderer.h"
#include "SceneGraph.h"
#include "GlobalIlluminationRenderer_IV.h"

#ifdef USING_GLUT	// defined in DeferredRenderer.h
	#include <GL/glut.h>
#endif

int		width, height; // window w/h
int     cur_preview=0;
int     dr_scale = 1;
bool    show_help = true;
float	helptime = 0.f;
bool    enable_preview = false;
bool    enable_stats = true;
bool    inj_camera = true;
bool    inj_lights = true;
bool    exiting = false;

World3D * world;
DeferredRenderer * dr;

void showHelp ();

void Idle()
{
	if (exiting)
	{
		delete world;
		delete dr;
	    exit (EXIT_SUCCESS);
	}

	static bool helpfirst = true;
	helptime = glutGet(GLUT_ELAPSED_TIME)/1000.0f;
	if (helpfirst && helptime > 10)	// disable help messages after ... secs
	{
		show_help = false;
		helptime = 0.f;
		helpfirst = false;
	}

	world->app();
	world->postApp ();
	world->cull ();

	glutPostRedisplay();
}

void RenderMainWindow()
{
	dr->draw();
	
	if (enable_preview)
	{
		glViewport(0,0,width,height);
		dr->show(cur_preview);	
	}

	if (show_help)
		showHelp ();

	GlobalIlluminationRendererIV * gi_iv = dynamic_cast<GlobalIlluminationRendererIV*>(dr->getGIRenderer());
	if (gi_iv != NULL)
		gi_iv->setInjectCameraLights (inj_camera, inj_lights);

	if (enable_stats)
		dr->showStatistics ();

	glutSwapBuffers();
}

void ResizeMainWindow( int x, int y )
{
	if (exiting)
		return;
	glViewport(0,0,x,y);
	width = x;
	height = y;
	dr->resize(width,height);
	glutPostRedisplay();
}

void SetupMainWindow () 
{ 
    glEnable(GL_DEPTH_TEST );
    glEnable(GL_ALPHA_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glClearColor(0.0,0.0,0.0,1);
    glDisable(GL_CULL_FACE);
#ifdef WIN32
	wglSwapIntervalEXT(0);
#endif
    glPolygonMode (GL_FRONT_AND_BACK,GL_FILL);
    glEnable(GL_COLOR_MATERIAL);
}

void Keyboard (unsigned char k, int x, int y)
{
	float range;

	switch (k)
	{
	case '1': cur_preview = DR_TARGET_COLOR;		break;
	case '2': cur_preview = DR_TARGET_NORMAL;		break;
	case '3': cur_preview = DR_TARGET_SPECULAR;		break;
	case '4': cur_preview = DR_TARGET_LIGHTING;		break;
	case '5': cur_preview = DR_TARGET_DEPTH;		break;
	case '6': cur_preview = DR_TARGET_FRAMEBUFFER;	break;
	case '7': cur_preview = DR_TARGET_SHADOWMAP;	break;
	case '8': cur_preview = DR_TARGET_GLOW;			break;
	case '9': cur_preview = DR_TARGET_AO;			break;
	case '0': cur_preview = DR_TARGET_VBO;			break;

	case ',': range=dr->getGIRenderer()->getFactor(); range*=0.8; printf("gi factor: %f\n", range); dr->getGIRenderer()->setFactor(range); break;
	case '.': range=dr->getGIRenderer()->getFactor(); range*=1.2; printf("gi factor: %f\n", range); dr->getGIRenderer()->setFactor(range); break;
	case 'n': range=dr->getGIRenderer()->getRange();  range*=0.8; printf("gi range: %f\n", range);  dr->getGIRenderer()->setRange(range); break;
	case 'm': range=dr->getGIRenderer()->getRange();  range*=1.2; printf("gi range: %f\n", range);  dr->getGIRenderer()->setRange(range); break;

	case 'C': case 'c': inj_camera = !inj_camera; break;
	case 'L': case 'l': inj_lights = !inj_lights; break;
	case 'P': case 'p': enable_preview = !enable_preview; break;
	case 'S': case 's': enable_stats = !enable_stats; break;
    case 'G': case 'g': dr->enableGI(!dr->isGIEnabled()); break;
	case 'H': case 'h': show_help = !show_help; break;
	case  27: exiting = true; break;
	}
}

void DisplayMessageBitmap(char * msg, int len, int x, int y)
{
	glEnable(GL_COLOR_MATERIAL);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_3D);
	glDisable(GL_LIGHTING);

	glColor3f(1,1,1);
	glLineWidth(1.0f);
	glPushMatrix();
		glRasterPos3f (x, y, 0);
		for (int i = 0; i<len; i++)
			glutBitmapCharacter(GLUT_BITMAP_9_BY_15, msg[i]);
	glPopMatrix();
	glDisable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);
}

void showHelp ()
{
	glBindFramebuffer(GL_FRAMEBUFFER_EXT, 0);
	glDrawBuffer(GL_BACK);
	glActiveTextureARB(GL_TEXTURE0);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glBindTexture(GL_TEXTURE_2D,0);
    glDisable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, width, 0.0, height, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	
	glColor4f(0.3f, 0.3f, 0.4f, 0.5f);
	float bl = width-475, br = width-10, bt=100, bb=10;
	glBegin(GL_QUADS);
	glVertex3f(bl,bt,0.5);
	glVertex3f(bl,bb,0.5);
	glVertex3f(br,bb,0.5);
	glVertex3f(br,bt,0.5);
	glEnd();

	char message[80];
	
	sprintf(message,"Navigation: Arrow Keys");
	DisplayMessageBitmap(message, strlen(message), bl+10, bt-20);
	
	sprintf(message,"Preview Buffers: 'P/p' followed by Number Keys");
	DisplayMessageBitmap(message, strlen(message), bl+10, bt-40);
	
	sprintf(message,"'C/c' 'L/l': Toggle injection of Camera or Lights");
	DisplayMessageBitmap(message, strlen(message), bl+10, bt-60);
	
	sprintf(message,"'H/h': Toggle help");
	DisplayMessageBitmap(message, strlen(message), bl+10, bt-80);
	
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	
	glEnable(GL_DEPTH_TEST);
}

int main(int argc, char ** argv)
{
	printf (
		"////////////////////////////////////////////////////////////////////////////// \n"
		"//                                                                          // \n"
		"//  Progressive Voxelization for Global Illumination Demo                   // \n"
		"//  demostrating both dynamic geometry and dynamic illumination             // \n"
		"//    Athanasios Gaitatzes (gaitat at yahoo dot com), 2012                  // \n"
		"//    Georgios Papaioannou (gepap at aueb dot gr), 2012                     // \n"
		"//                                                                          // \n"
		"//  If you use this code as is or any part of it in any kind                // \n"
		"//  of project or product, please acknowledge the source and its authors    // \n"
		"//  by citing their work:                                                   // \n"
		"//                                                                          // \n"
		"//    A. Gaitatzes, G. Papaioannou,                                         // \n"
		"//    \"Progressive Screen-space Multi-channel Surface Voxelization\"         // \n" 
		"//                                                                          // \n"
		"//  You can find the latest version of the code at:                         // \n"
		"//  http://www.virtuality.gr/gaitat/en/publications.html                    // \n"
		"//                                                                          // \n"
		"////////////////////////////////////////////////////////////////////////////// \n\n");

	glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE|GLUT_DEPTH);
	glutInitWindowSize (960,540); // (768,432); // (640,640); // 
	glutInitWindowPosition (640,220);
	glutCreateWindow ("Progressive Voxelization Demo");
	glutDisplayFunc(RenderMainWindow);
	glutReshapeFunc(ResizeMainWindow);
	glutKeyboardFunc (Keyboard);
	glutIdleFunc(Idle);

#ifdef WIN32
    // init the OpenGL extensions (required on Windows platform)
	glewInit();
#endif
    SetupMainWindow();
	glutSetCursor(GLUT_CURSOR_NONE);

    dr = new DeferredRenderer();
	if (dr->init() != DR_ERROR_NONE)
		EAZD_TRACE ("SceneViewer : ERROR - renderer not properly initialized.");
	dr->setBufferScale(1.0);

	world = new World3D();
	world->setRenderer(dr);
	world->load (STR_DUP (argv[1]));
	world->init();
	world->preApp();
	
	glutMainLoop();
	
	delete world;
	delete dr;

	return 0;
}
