#include "canvas.h"
#include <GL/glut.h>

#include <iostream>

Canvas* canvas = NULL;

void motion(int x, int y) {
	canvas->mouseInputHandler().mouseMoveEvent(x, y);
}

void mouse(int button, int state, int x, int y) {
	if(canvas != NULL) {
		if(state==GLUT_DOWN) canvas->mouseInputHandler().mousePressEvent(button, x, y);
		else if(state==GLUT_UP) canvas->mouseInputHandler().mouseReleaseEvent(button, x, y);
	}
}

void idle() {
	if(canvas!=0 && canvas->idleRedraw()) glutPostRedisplay();
}

void keys( unsigned char c, int x, int y) {
    switch (c) {
        case 27: //escape key
            exit(0);
            break;

        default:
            if(canvas != NULL) canvas->keyPressEvent(c, x, y);
            break;
    }

    glutPostRedisplay();
}

void reshape(int w, int h) {
    if(canvas != NULL) canvas->resizeGL(w, h);
}

void display() {
    if(canvas != NULL) canvas->paintGL();

    glutSwapBuffers();
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayString("double rgb~8 depth~24 samples<=8");
    int width, height;
    //width = 2048;
    //height = 1024;
    width = 1280;
    height = 960;
    //width = 1024;
    //height = 768;
    glutInitWindowSize(width, height);
    glutCreateWindow("SSBentConesDemo");

    canvas = new Canvas(width, height);

    glewInit();

    if(!(glewIsSupported("GL_VERSION_3_3"))) {
        printf("Unable to load extensions\n\nExiting...\n");
        exit(-1);
    }

    canvas->initializeGL();

    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion); 
    glutIdleFunc(idle);

	std::cout << "Usage:" << std::endl;
	std::cout << "  Navigation:" << std::endl;
	std::cout << "    Left mouse button:       look around" << std::endl;
	std::cout << "    Middle mouse button:     examine" << std::endl;
	std::cout << "    w, a, s, d:			   move around horizontally" << std::endl;
	std::cout << "    q, e:                    move up/down" << std::endl;
	std::cout << "    +, -:                    zoom" << std::endl;
	std::cout << std::endl;
	std::cout << "  Options:" << std::endl;
	std::cout << "    1:       half AO world-space radius" << std::endl;
	std::cout << "    2:       double AO world-space radius" << std::endl;
	std::cout << "    3:       half AO sample count" << std::endl;
	std::cout << "    4:       double AO sample count" << std::endl;
	std::cout << "    5:       -1 ray marching steps per AO sample" << std::endl;
	std::cout << "    6:       +1 ray marching steps per AO sample" << std::endl;
	std::cout << "    t:       toggle timers" << std::endl;
	std::cout << "  Exit: ESC" << std::endl << std::endl << std::endl;

    glutMainLoop();
    return 0;
}


