#ifdef WIN32
	#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <SDL/SDL.h>

#include <node.hpp>



int screenWidth = 800;
int screenHeight = 600;

bool keys[512];
int mouseX, mouseY, mouseRelX, mouseRelY;

mtx viewProjTransform;



void OnKeyDown(int key);



void Quit(int code)
{
	SDL_Quit();
	exit(code);
}



void ProcessEvents()
{
	SDL_Event event;
	static bool firstMouseMotionEventProcessed = false;

	mouseRelX = 0;
	mouseRelY = 0;

	while (SDL_PollEvent(&event))
	{
		switch(event.type)
		{
		case SDL_KEYDOWN:
			keys[event.key.keysym.sym] = true;
			OnKeyDown(event.key.keysym.sym);
			break;

		case SDL_KEYUP:
			keys[event.key.keysym.sym] = false;
			break;

		case SDL_MOUSEMOTION:
			mouseX = event.motion.x;
			mouseY = event.motion.y;
			if (firstMouseMotionEventProcessed)
			{
				mouseRelX = event.motion.xrel;
				mouseRelY = event.motion.yrel;
			}
			firstMouseMotionEventProcessed = true;
			break;

		case SDL_QUIT:
			Quit(0);
			break;
		}
	}
}



void RenderUnitCube()
{
	// front
	glBegin(GL_POLYGON);
	glVertex3f(  0.5, -0.5, 0.5 );
	glVertex3f(  0.5,  0.5, 0.5 );
	glVertex3f( -0.5,  0.5, 0.5 );
	glVertex3f( -0.5, -0.5, 0.5 );
	glEnd();

	// back
	glBegin(GL_POLYGON);
	glVertex3f(  0.5, -0.5, -0.5 );
	glVertex3f(  0.5,  0.5, -0.5 );
	glVertex3f( -0.5,  0.5, -0.5 );
	glVertex3f( -0.5, -0.5, -0.5 );
	glEnd();

	// right
	glBegin(GL_POLYGON);
	glVertex3f( 0.5, -0.5, -0.5 );
	glVertex3f( 0.5,  0.5, -0.5 );
	glVertex3f( 0.5,  0.5,  0.5 );
	glVertex3f( 0.5, -0.5,  0.5 );
	glEnd();

	// left
	glBegin(GL_POLYGON);
	glVertex3f( -0.5, -0.5,  0.5 );
	glVertex3f( -0.5,  0.5,  0.5 );
	glVertex3f( -0.5,  0.5, -0.5 );
	glVertex3f( -0.5, -0.5, -0.5 );
	glEnd();

	// top
	glBegin(GL_POLYGON);
	glVertex3f(  0.5,  0.5,  0.5 );
	glVertex3f(  0.5,  0.5, -0.5 );
	glVertex3f( -0.5,  0.5, -0.5 );
	glVertex3f( -0.5,  0.5,  0.5 );
	glEnd();

	// bottom
	glBegin(GL_POLYGON);
	glVertex3f(  0.5, -0.5, -0.5 );
	glVertex3f(  0.5, -0.5,  0.5 );
	glVertex3f( -0.5, -0.5,  0.5 );
	glVertex3f( -0.5, -0.5, -0.5 );
	glEnd();
}



void RenderObject(const mtx& globalTransform, const vec3& color)
{
	mtx localTransform, worldViewProjTransform;

	//

	localTransform = globalTransform;
	worldViewProjTransform = localTransform * viewProjTransform;

	glLoadIdentity();
	glLoadMatrixf(&worldViewProjTransform._[0][0]);
	glColor3f(color.x, color.y, color.z);
	RenderUnitCube();

	//

	localTransform = mtx::Scale(0.5f, 0.5f, 0.5f) * mtx::Translate(0.75f, 0.0f, 0.0f) * globalTransform;
	worldViewProjTransform = localTransform * viewProjTransform;

	glLoadIdentity();
	glLoadMatrixf(&worldViewProjTransform._[0][0]);
	glColor3f(1.0f, 1.0f, 1.0f);
	RenderUnitCube();
}



Node cameraNode;

const int objectsNodesNum = 4;
Node objectsNodes[objectsNodesNum];
string objectsNodesNames[objectsNodesNum] =
	{
		"Red",
		"Green",
		"Blue",
		"Grey"
	};
vec3 objectsNodesColors[objectsNodesNum] =
	{
		vec3(1.0f, 0.0f, 0.0f),
		vec3(0.0f, 1.0f, 0.0f),
		vec3(0.0f, 0.0f, 1.0f),
		vec3(0.5f, 0.5f, 0.5f),
	};



void InitObjectsNodes(int configurationIndex)
{
	for (int i = 0; i < objectsNodesNum; i++)
	{
		objectsNodes[i].name = objectsNodesNames[i];
		objectsNodes[i].SetParent(0); // "reset" parent
		objectsNodes[i].SetLocalTranslation(0.0f, 0.0f, 0.0f);
		objectsNodes[i].SetLocalRotation(quat::Identity());
		objectsNodes[i].SetLocalScale(0.0f, 0.0f, 0.0f); // make all invisible
	}

	if (configurationIndex == 0)
	{
		objectsNodes[0].SetChild(&objectsNodes[1]);

		objectsNodes[0].SetLocalTranslation(-8.0f, 0.0f, 0.0f);
		objectsNodes[0].SetLocalScale(4.0f, 1.0f, 1.0f);

		objectsNodes[1].SetLocalTranslation(2.0f, 0.0f, 0.0f);
		objectsNodes[1].SetLocalRotation(quat::RotateY(pi/4.0f));
		objectsNodes[1].SetLocalScale(0.5f, 0.5f, 0.5f);
	}
	else if (configurationIndex == 1)
	{
		objectsNodes[0].SetLocalTranslation(-7.5f, 0.0f, 0.0f);
		objectsNodes[0].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[1].SetLocalTranslation(-2.5f, 0.0f, 0.0f);
		objectsNodes[1].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[2].SetLocalTranslation(2.5f, 0.0f, 0.0f);
		objectsNodes[2].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[3].SetLocalTranslation(7.5f, 0.0f, 0.0f);
		objectsNodes[3].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[0].SetChild(&objectsNodes[1]);
		objectsNodes[1].SetChild(&objectsNodes[2]);
		objectsNodes[2].SetChild(&objectsNodes[3]);
	}
	else if (configurationIndex == 2)
	{
		objectsNodes[0].SetChild(&objectsNodes[1]);
		objectsNodes[1].SetChild(&objectsNodes[2]);
		objectsNodes[2].SetChild(&objectsNodes[3]);

		objectsNodes[0].SetLocalTranslation(-7.5f, 0.0f, 0.0f);
		objectsNodes[0].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[1].SetLocalTranslation(-2.5f, 0.0f, 0.0f);
		objectsNodes[1].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[2].SetLocalTranslation(2.5f, 0.0f, 0.0f);
		objectsNodes[2].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[3].SetLocalTranslation(7.5f, 0.0f, 0.0f);
		objectsNodes[3].SetLocalScale(1.0f, 1.0f, 1.0f);
	}
	else if (configurationIndex == 3)
	{
		objectsNodes[0].SetLocalTranslation(-7.5f, 0.0f, 0.0f);
		objectsNodes[0].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[1].SetLocalTranslation(-2.5f, 0.0f, 0.0f);
		objectsNodes[1].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[2].SetLocalTranslation(2.5f, 0.0f, 0.0f);
		objectsNodes[2].SetLocalScale(1.0f, 1.0f, 1.0f);

		objectsNodes[3].SetLocalTranslation(7.5f, 0.0f, 0.0f);
		objectsNodes[3].SetLocalScale(1.0f, 1.0f, 1.0f);
	}
}



void Init()
{
	glEnable(GL_DEPTH_TEST);

	cameraNode.SetLocalTranslation(0.0f, 0.0f, 10.0f);

	InitObjectsNodes(0);

	cout << "F1 - objects configuration 1" << endl;
	cout << "F2 - objects configuration 2" << endl;
	cout << "F3 - objects configuration 3" << endl;
	cout << "F4 - objects configuration 4" << endl;
	cout << "1 - red object" << endl;
	cout << "2 - green object" << endl;
	cout << "3 - blue object" << endl;
	cout << "4 - grey object" << endl;
}



void Update(float deltaTime)
{
	glViewport(0, 0, screenWidth, screenHeight);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// camera
	{
		float speed = 0.02f;
		if (keys[SDLK_LSHIFT])
			speed = 0.08f;

		if (keys[SDLK_w])
			cameraNode.SetLocalTranslation(cameraNode.GetLocalTranslation() + speed*deltaTime*cameraNode.GetLocalForward());
		if (keys[SDLK_s])
			cameraNode.SetLocalTranslation(cameraNode.GetLocalTranslation() - speed*deltaTime*cameraNode.GetLocalForward());
		if (keys[SDLK_a])
			cameraNode.SetLocalTranslation(cameraNode.GetLocalTranslation() - speed*deltaTime*cameraNode.GetLocalRight());
		if (keys[SDLK_d])
			cameraNode.SetLocalTranslation(cameraNode.GetLocalTranslation() + speed*deltaTime*cameraNode.GetLocalRight());

		cameraNode.LocalRotate(quat::RotateY(-mouseRelX / 1000.0f));
		cameraNode.LocalRotate(quat::Rotate(cameraNode.GetLocalRight(), -mouseRelY / 1000.0f));

		//

		vec3 globalTranslation = cameraNode.GetGlobalTranslation();
		vec3 right, up, backward;
		cameraNode.ComputeGlobalBasisVectors(right, up, backward);

		viewProjTransform =
			Matrix::LookAtRH(globalTranslation, globalTranslation - backward, up) *
			Matrix::PerspectiveFovRH(pi/3.0f, (float)screenWidth/(float)screenHeight, 1.0f, 100.0f);
	}

	// objects
	{
		for (int i = 0; i < objectsNodesNum; i++)
			RenderObject(objectsNodes[i].GetGlobalMatrix(), objectsNodesColors[i]);
	}
}



void OnKeyDown(int key)
{
	if (keys[SDLK_F1])
		InitObjectsNodes(0);
	if (keys[SDLK_F2])
		InitObjectsNodes(1);
	if (keys[SDLK_F3])
		InitObjectsNodes(2);
	if (keys[SDLK_F4])
		InitObjectsNodes(3);

	if (keys[SDLK_1] || keys[SDLK_2] || keys[SDLK_3] || keys[SDLK_4])
	{
		int objectIndex = 0;
		if (keys[SDLK_2]) objectIndex = 1;
		if (keys[SDLK_3]) objectIndex = 2;
		if (keys[SDLK_4]) objectIndex = 3;

		cout << endl;
		cout << "h - hierarchy" << endl;
		cout << "t - local translation" << endl;
		cout << "r - local rotation (euler angles, in degrees)" << endl;
		cout << "s - local scale" << endl;
		cout << "p - set parent index" << endl;
		cout << "c - set child index" << endl;

		char operation = getchar();

		if (operation == 'h')
		{
			objectsNodes[objectIndex].Debug_Recursive(&objectsNodes[objectIndex], 0);
		}
		else if (operation == 's')
		{
			float x, y, z;

			cout << "x = " << objectsNodes[objectIndex].GetLocalScale().x << endl;
			cout << "y = " << objectsNodes[objectIndex].GetLocalScale().y << endl;
			cout << "z = " << objectsNodes[objectIndex].GetLocalScale().z << endl;

			cout << "x = ";
			cin >> x;
			cout << "y = ";
			cin >> y;
			cout << "z = ";
			cin >> z;

			objectsNodes[objectIndex].SetLocalScale(x, y, z);
		}
		else if (operation == 'r')
		{
			float x, y, z;

			cout << "x = " << RadToDeg(objectsNodes[objectIndex].GetLocalEulerAngles().x) << endl;
			cout << "y = " << RadToDeg(objectsNodes[objectIndex].GetLocalEulerAngles().y) << endl;
			cout << "z = " << RadToDeg(objectsNodes[objectIndex].GetLocalEulerAngles().z) << endl;

			cout << "x = ";
			cin >> x;
			cout << "y = ";
			cin >> y;
			cout << "z = ";
			cin >> z;

			objectsNodes[objectIndex].SetLocalEulerAngles(DegToRad(x), DegToRad(y), DegToRad(z));
		}
		else if (operation == 't')
		{
			float x, y, z;

			cout << "x = " << objectsNodes[objectIndex].GetLocalTranslation().x << endl;
			cout << "y = " << objectsNodes[objectIndex].GetLocalTranslation().y << endl;
			cout << "z = " << objectsNodes[objectIndex].GetLocalTranslation().z << endl;

			cout << "x = ";
			cin >> x;
			cout << "y = ";
			cin >> y;
			cout << "z = ";
			cin >> z;

			objectsNodes[objectIndex].SetLocalTranslation(x, y, z);
		}
		else if (operation == 'p')
		{
			int index;

			cout << "parent index (0 means no parent) = ";
			cin >> index;

			if (index == 0)
				objectsNodes[objectIndex].SetParent(NULL);
			else if (index >= 1 && index <= objectsNodesNum)
				objectsNodes[objectIndex].SetParent(&objectsNodes[index - 1]);
		}
		else if (operation == 'c')
		{
			int index;

			cout << "child index = ";
			cin >> index;

			if (index >= 1 && index <= objectsNodesNum)
				objectsNodes[objectIndex].SetChild(&objectsNodes[index - 1]);
		}
	}
}



#ifdef WIN32
#undef main
#endif
int main(int argc, char* argv[])
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		fprintf(stderr, "Video initialization failed: %s\n", SDL_GetError());
		Quit(1);
	}

	SDL_putenv((char*)"SDL_VIDEO_CENTERED=1");

	const SDL_VideoInfo* info = SDL_GetVideoInfo();

	if (!info)
	{
		fprintf(stderr, "Video query failed: %s\n", SDL_GetError());
		Quit(1);
	}

	if (SDL_SetVideoMode(screenWidth, screenHeight, info->vfmt->BitsPerPixel, SDL_OPENGL) == 0)
	{
		fprintf(stderr, "Video mode set failed: %s\n", SDL_GetError());
		Quit(1);
	}

	SDL_ShowCursor(false);
	SDL_WM_GrabInput((SDL_GrabMode)true);

	Init();
	int ticksBefore, ticksAfter;
	int lastFrameTime = 0;

	while (1)
	{
		ticksBefore = SDL_GetTicks();

		ProcessEvents();
		Update((float)lastFrameTime);
		SDL_GL_SwapBuffers();

		ticksAfter = SDL_GetTicks();
		lastFrameTime = ticksAfter - ticksBefore;

		if (keys[SDLK_ESCAPE])
			Quit(0);
	}

	return 0;
}
