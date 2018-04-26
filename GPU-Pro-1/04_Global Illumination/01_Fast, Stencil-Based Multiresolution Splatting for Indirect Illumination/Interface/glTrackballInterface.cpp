/***************************************************************************/
/* glTrackballInterface.cpp                                                */
/* -----------------------                                                 */
/*                                                                         */
/* Defines a series of functions used in glInterface.cpp for updating the  */
/*    trackballs using GLUT/OpenGL feedback on the current mouse position. */
/*                                                                         */
/* These could go in glInterface, but they're messy and shouldn't need to  */
/*    change (unless the # of object/light trackballs allowed increases.)  */
/*                                                                         */
/* Chris Wyman (02/01/2008)                                                */
/***************************************************************************/

#include "stencilMultiResSplatting.h"
#include "Scene/Scene.h"
#include "Interface/SceneFileDefinedInteraction.h"

extern SceneDefinedUI *ui;

extern Trackball *eyeBall;
extern Trackball *lightBall[4];
extern Trackball *objBall[4];


void UpdateTrackball( int curCommand, int xVal, int yVal )
{
	if (curCommand == ui->Key( UI_EYE_TRACKBALL ) && eyeBall)
		eyeBall->UpdateTrackballOnMotion(xVal,yVal);
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_0 ) && lightBall[0])
		lightBall[0]->UpdateTrackballOnMotion(xVal,yVal);
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_1 ) && lightBall[1])
		lightBall[1]->UpdateTrackballOnMotion(xVal,yVal);
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_2 ) && lightBall[2])
		lightBall[2]->UpdateTrackballOnMotion(xVal,yVal);
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_3 ) && lightBall[3])
		lightBall[3]->UpdateTrackballOnMotion(xVal,yVal);
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_0 ) && objBall[0])
		objBall[0]->UpdateTrackballOnMotion(xVal,yVal);
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_1 ) && objBall[1])
		objBall[1]->UpdateTrackballOnMotion(xVal,yVal);
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_2 ) && objBall[2])
		objBall[2]->UpdateTrackballOnMotion(xVal,yVal);
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_3 ) && objBall[3])
		objBall[3]->UpdateTrackballOnMotion(xVal,yVal);
}

void InitializeTrackball( int curCommand, int xVal, int yVal )
{
	if (curCommand == ui->Key( UI_EYE_TRACKBALL ) && eyeBall)
		eyeBall->SetTrackballOnClick(xVal,yVal);
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_0 ) && lightBall[0])
		lightBall[0]->SetTrackballOnClick(xVal,yVal);
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_1 ) && lightBall[1])
		lightBall[1]->SetTrackballOnClick(xVal,yVal);
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_2 ) && lightBall[2])
		lightBall[2]->SetTrackballOnClick(xVal,yVal);
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_3 ) && lightBall[3])
		lightBall[3]->SetTrackballOnClick(xVal,yVal);
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_0 ) && objBall[0])
		objBall[0]->SetTrackballOnClick(xVal,yVal);
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_1 ) && objBall[1])
		objBall[1]->SetTrackballOnClick(xVal,yVal);
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_2 ) && objBall[2])
		objBall[2]->SetTrackballOnClick(xVal,yVal);
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_3 ) && objBall[3])
		objBall[3]->SetTrackballOnClick(xVal,yVal);
}

void ReleaseTrackball( int curCommand )
{
	if (curCommand == ui->Key( UI_EYE_TRACKBALL ) && eyeBall)
		eyeBall->ReleaseTrackball();
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_0 ) && lightBall[0])
		lightBall[0]->ReleaseTrackball();
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_1 ) && lightBall[1])
		lightBall[1]->ReleaseTrackball();
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_2 ) && lightBall[2])
		lightBall[2]->ReleaseTrackball();
	else if (curCommand == ui->Key( UI_LIGHT_TRACKBALL_3 ) && lightBall[3])
		lightBall[3]->ReleaseTrackball();
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_0 ) && objBall[0])
		objBall[0]->ReleaseTrackball();
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_1 ) && objBall[1])
		objBall[1]->ReleaseTrackball();
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_2 ) && objBall[2])
		objBall[2]->ReleaseTrackball();
	else if (curCommand == ui->Key( UI_OBJECT_TRACKBALL_3 ) && objBall[3])
		objBall[3]->ReleaseTrackball();
}