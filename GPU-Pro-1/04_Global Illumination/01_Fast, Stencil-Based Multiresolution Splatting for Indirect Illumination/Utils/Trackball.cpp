/* trackball.c                               */
/* -----------                               */
/*                                           */
/* Code to implement a simple trackball-like */
/*     motion control.                       */
/*                                           */
/* This expands on Ed Angel's trackball.c    */
/*     demo program.  Though I think I've    */
/*     seen this code (trackball_ptov)       */
/*     before elsewhere.                     */
/*********************************************/


#include "trackball.h"
#include <math.h>

#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif 

void matrixIdentity(float m[16])
{
    m[0+4*0] = 1; m[0+4*1] = 0; m[0+4*2] = 0; m[0+4*3] = 0;
    m[1+4*0] = 0; m[1+4*1] = 1; m[1+4*2] = 0; m[1+4*3] = 0;
    m[2+4*0] = 0; m[2+4*1] = 0; m[2+4*2] = 1; m[2+4*3] = 0;
    m[3+4*0] = 0; m[3+4*1] = 0; m[3+4*2] = 0; m[3+4*3] = 1;
}

int matrixInvert(float src[16], float inverse[16])
{
    float t;
    int i, j, k, swap;
    float tmp[4][4];

    matrixIdentity(inverse);

    for (i = 0; i < 4; i++) {
	for (j = 0; j < 4; j++) {
	    tmp[i][j] = src[i*4+j];
	}
    }

    for (i = 0; i < 4; i++) {
        /* look for largest element in column. */
        swap = i;
        for (j = i + 1; j < 4; j++) {
            if (fabs(tmp[j][i]) > fabs(tmp[i][i])) {
                swap = j;
            }
        }

        if (swap != i) {
            /* swap rows. */
            for (k = 0; k < 4; k++) {
                t = tmp[i][k];
                tmp[i][k] = tmp[swap][k];
                tmp[swap][k] = t;

                t = inverse[i*4+k];
                inverse[i*4+k] = inverse[swap*4+k];
                inverse[swap*4+k] = t;
            }
        }

        if (tmp[i][i] == 0) {
            /* no non-zero pivot.  the matrix is singular, which
	       shouldn't happen.  This means the user gave us a bad
	       matrix. */
            return 0;
        }

        t = tmp[i][i];
        for (k = 0; k < 4; k++) {
            tmp[i][k] /= t;
            inverse[i*4+k] /= t;
        }
        for (j = 0; j < 4; j++) {
            if (j != i) {
                t = tmp[j][i];
                for (k = 0; k < 4; k++) {
                    tmp[j][k] -= tmp[i][k]*t;
                    inverse[j*4+k] -= inverse[i*4+k]*t;
                }
            }
        }
    }
    return 1;
}

TrackballList::TrackballList( int num, int width, int height )
{
	list = (Trackball **) malloc ( num * sizeof( Trackball *) );
	if (!list) printf("Error! Unable to allocate memory for trackball list!\n");
	listSize = num;

	for (int i = 0; i < num; i++)
		list[i] = new Trackball( width, height );
}

/* resizes all the trackballs in a list */
void TrackballList::ResizeTrackballs( int width, int height )
{
	for (int i = 0; i < listSize; i++)
		list[i]->ResizeTrackballWindow( width, height );
}

/* sets the size of the window the trackball assumes */
void Trackball::ResizeTrackballWindow( int width, int height )
{
	ballWidth = width;
	ballHeight = height;
}

/* the internal code which computes the rotation */
void Trackball::trackball_ptov(int x, int y, int width, int height, float v[3])
{
    float d, a;

    /* project x,y onto a hemi-sphere centered within width, height */
    v[0] = (2.0F*x - width) / width;
    v[1] = (height - 2.0F*y) / height;
    d = (float) sqrt(v[0]*v[0] + v[1]*v[1]);
    v[2] = (float) cos((M_PI/2.0F) * ((d < 1.0F) ? d : 1.0F));
    a = 1.0F / (float) sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] *= a;
    v[1] *= a;
    v[2] *= a;
}

void Trackball::SetTrackballOnClick( int x, int y )
{
	trackball_ptov( x, y, ballWidth, ballHeight, lastPos );
	currentlyTracking = 1;
}

void Trackball::UpdateTrackballOnMotion( int x, int y )
{
	float curPos[3], dx, dy, dz, angle, axis[3];
	if (!currentlyTracking) return;

	trackball_ptov( x, y, ballWidth, ballHeight, curPos );
    dx = curPos[0] - lastPos[0];
	dy = curPos[1] - lastPos[1];
	dz = curPos[2] - lastPos[2];
	if ( fabs(dx) > 0 || fabs(dy) > 0 || fabs(dz) > 0 )
	{
		angle = 90 * sqrt( dx*dx + dy*dy + dz*dz );
		axis[0] = lastPos[1]*curPos[2] - lastPos[2]*curPos[1];
		axis[1] = lastPos[2]*curPos[0] - lastPos[0]*curPos[2];
		axis[2] = lastPos[0]*curPos[1] - lastPos[1]*curPos[0];
		lastPos[0] = curPos[0];
		lastPos[1] = curPos[1];
		lastPos[2] = curPos[2];
		glPushMatrix();
		glLoadIdentity();
        glRotatef( angle, axis[0], axis[1], axis[2] );
		glMultMatrixf( trackballMatrix.GetDataPtr() );
		glGetFloatv( GL_MODELVIEW_MATRIX, trackballMatrix.GetDataPtr() );
		inverseTrackballMatrix = trackballMatrix.Invert();
		glPopMatrix();
		hasChanged = 1;
	}
}

Trackball::Trackball( int width, int height )
{
	ballWidth = width; 
	ballHeight = height;
	currentlyTracking = 0;
	lastPos[0] = lastPos[1] = lastPos[2] = 0; 
	hasChanged = 1;
}

void Trackball::ResetTrackball( void )
{
	currentlyTracking = 0;
	lastPos[0] = lastPos[1] = lastPos[2] = 0; 
	trackballMatrix = Matrix4x4::Identity();
	inverseTrackballMatrix = Matrix4x4::Identity();
	hasChanged = 1;
}

void Trackball::MultiplyTrackballMatrix( void )
{
	glMultMatrixf( trackballMatrix.GetDataPtr() );
}

void Trackball::PrintTrackballMatrix( void )
{
	printf("Trackball Matrix:\n" );
	trackballMatrix.Print();
}


void Trackball::MultiplyTransposeTrackballMatrix( void )
{
	glMultTransposeMatrixf( trackballMatrix.GetDataPtr() );
}

void Trackball::MultiplyInverseTrackballMatrix( void )
{
	glMultMatrixf( inverseTrackballMatrix.GetDataPtr() );
}

void Trackball::MultiplyInverseTransposeTrackballMatrix( void )
{
	glMultTransposeMatrixf( inverseTrackballMatrix.GetDataPtr() );
}


void Trackball::ApplyTrackballMatrix( float inVec[4], float result[4] )
{
	result[0] = trackballMatrix[0]*inVec[0] + trackballMatrix[4]*inVec[1] + trackballMatrix[8]*inVec[2] + trackballMatrix[12]*inVec[3];
	result[1] = trackballMatrix[1]*inVec[0] + trackballMatrix[5]*inVec[1] + trackballMatrix[9]*inVec[2] + trackballMatrix[13]*inVec[3];
	result[2] = trackballMatrix[2]*inVec[0] + trackballMatrix[6]*inVec[1] + trackballMatrix[10]*inVec[2] + trackballMatrix[14]*inVec[3];
	result[3] = trackballMatrix[3]*inVec[0] + trackballMatrix[7]*inVec[1] + trackballMatrix[11]*inVec[2] + trackballMatrix[15]*inVec[3];
}

Vector Trackball::ApplyTrackballMatrix( const Vector &vec )
{
	return trackballMatrix * vec;
}

Point Trackball::ApplyTrackballMatrix( const Point &pt )
{
	return trackballMatrix * pt;
}

void Trackball::SetTrackballMatrix( GLfloat *newMat )
{
	trackballMatrix = Matrix4x4( newMat );
	inverseTrackballMatrix = trackballMatrix.Invert();
	hasChanged = 1;
}

void Trackball::SetTrackballMatrix( const Matrix4x4 &newMat )
{
	trackballMatrix = newMat;
	inverseTrackballMatrix = trackballMatrix.Invert();
	hasChanged = 1;
}


void Trackball::GetTrackBallMatrix( GLfloat *matrix )
{
	for (int i=0;i<16;i++) 
		matrix[i] = trackballMatrix[i];
}

void Trackball::GetInverseTrackBallMatrix( GLfloat *matrix )
{
	for (int i=0;i<16;i++) 
		matrix[i] = inverseTrackballMatrix[i];
}


bool Trackball::HasChanged( void )
{
	bool curVal = (hasChanged != 0);
	hasChanged = 0;
	return curVal;
}



