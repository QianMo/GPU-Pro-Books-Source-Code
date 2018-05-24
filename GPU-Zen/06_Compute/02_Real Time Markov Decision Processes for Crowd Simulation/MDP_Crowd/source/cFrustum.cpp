/*

Copyright 2013,2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901

Sergio Ruiz and Benjamin Hernandez. 2015. A Parallel Solver for Markov Decision Process
in Crowd Simulations. Fourteenth Mexican International Conference on Artificial
Intelligence (MICAI), Cuernavaca, 2015, pp. 107-116.
DOI: 10.1109/MICAI.2015.23

*/
#include "cFrustum.h"

//=======================================================================================

Frustum::Frustum( int type, float fov, float nearD, float farD )
{
	proj_man		= new ProjectionManager();
	culled_count	= 0;
	vfc_type		= type;
	fR				= (float)(rand() % 256) / 255.0f;
	fG				= (float)(rand() % 256) / 255.0f;
	fB				= (float)(rand() % 256) / 255.0f;
	RATIO			= proj_man->getAspectRatio();
	FOVY			= fov;
	TANG			= tan( DEG2RAD * FOVY * 0.5f );
	near_distance   = nearD;
	far_distance    = farD;
	near_height		= near_distance * TANG;
	near_width		= near_height   * RATIO;
	far_height		= far_distance  * TANG;
	far_width		= far_height    * RATIO;
	CAM_POS         = vec3( 0.0f );

	if( vfc_type == RADAR )
	{
		sphereFactorY = 1.0f / cos( FOVY );
		sphereFactorX = 1.0f / cos( atan( TANG * RATIO ) );
	}
}

//=======================================================================================

Frustum::~Frustum( void )
{
	FREE_INSTANCE( proj_man );
}

//=======================================================================================

int Frustum::pointInFrustum( vec3& p )
{
	if( vfc_type == GEOMETRIC )
	{
		int result = INSIDE;
		for( int i = 0; i < 6; i++ )
		{
			if( frustum_planes[i].distance( p ) < 0 )
			{
				return OUTSIDE;
			}
		}
		return result;
	}
	else if( vfc_type == RADAR )
	{
		float pcz, pcx, pcy, aux;
		// Compute vector from camera position to p.
		vec3 v = p - CAM_POS;
		// Compute and test the Z coordinate.
		pcz = dot( v, -Z );
		if( pcz > far_distance || pcz < near_distance )
		{
			return OUTSIDE;
		}
		// Compute and test the Y coordinate.
		pcy = dot( v, Y );
		aux = pcz * TANG;
		if( pcy > aux || pcy < -aux )
		{
			return OUTSIDE;
		}
		// Compute and test the X coordinate.
		pcx = dot( v, X );
		aux = aux * RATIO;
		if( pcx > aux || pcx < -aux )
		{
			return OUTSIDE;
		}
		return INSIDE;
	}
	else	// NO CPU VFC METHOD IS BEING USED.
	{
		return INSIDE;
	}
}

//=======================================================================================

int Frustum::sphereInFrustum( vec3& p, float radius )
{
	if( vfc_type == GEOMETRIC )
	{
		int result = INSIDE;
		float distance;
		for( int i = 0; i < 6; i++ )
		{
			distance = frustum_planes[i].distance( p );
			if( distance < -radius )
			{
				return OUTSIDE;
			}
			else if( distance < radius )
			{
				result = INTERSECT;
			}
		}
		return result;
	}
	else if( vfc_type == RADAR )
	{
		float d1, d2;
		float az, ax, ay, zz1, zz2;
		int result = INSIDE;
		vec3 v = p - CAM_POS;
		az = dot( v, -Z );
		if( az > far_distance + radius || az < near_distance - radius )
		{
			return OUTSIDE;
		}
		ax = dot( v, X );
		zz1 = az * TANG * RATIO;
		d1 = sphereFactorX * radius;
		if( ax > zz1 + d1 || ax < -zz1 - d1 )
		{
			return OUTSIDE;
		}
		ay = dot( v, Y );
		zz2 = az * TANG;
		d2 = sphereFactorY * radius;
		if( ay > zz2 + d2 || ay < -zz2 - d2 )
		{
			return OUTSIDE;
		}
		if( az > far_distance - radius || az < near_distance + radius )
		{
			result = INTERSECT;
		}
		if( ay > zz2 - d2 || ay < -zz2 + d2 )
		{
			result = INTERSECT;
		}
		if( ax > zz1 - d1 || ax < -zz1 + d1 )
		{
			result = INTERSECT;
		}
		return result;
	}
	else	// NO CPU VFC METHOD IS BEING USED
	{
		return INSIDE;
	}
}

//=======================================================================================

int Frustum::boxInFrustum( vec3& box_center, vec3& box_halfdiag )
{
	if( vfc_type != NONE )
	{
		int outCount = 0;
		for( int i = 0; i < 6; i++ )
		{
			float extents = box_halfdiag.x * abs( frustum_planes[i].N.x ) +
							box_halfdiag.y * abs( frustum_planes[i].N.y ) +
							box_halfdiag.z * abs( frustum_planes[i].N.z );
			float signed_distance = frustum_planes[i].distance( box_center );
			if( (signed_distance + extents) < 0 )		// BOX IS INSIDE PLANE =>
			{											// BOX IS OUTSIDE FRUSTUM.
				return OUTSIDE;
			}
			else if( (signed_distance - extents) > 0 )	// BOX IS OUTSIDE PLANE.
			{
				outCount++;
			}
		}
		if( outCount == 6 )								// BOX IS OUTSIDE ALL PLANES =>
		{												// BOX IS INSIDE FRUSTUM.
			return INSIDE;
		}
		else
		{
			return INTERSECT;
		}
	}
	else	// NO CPU VFC METHOD IS BEING USED
	{
		return INSIDE;
	}
}

//=======================================================================================

void Frustum::setPlanes( vec3& P, vec3& L, vec3& U )
{
	// No need to set planes if no CPU VFC is performed:
	if( vfc_type != NONE )
	{
		CAM_POS = P;
		Z = glm::normalize( CAM_POS - L );
		X = glm::normalize( glm::cross( U, Z ) );
		Y = glm::cross( Z, X );
		near_center = CAM_POS - (near_distance * Z);
		far_center  = CAM_POS - (far_distance  * Z);

		// Compute the 8 frustum corners:
		NTL = near_center + (near_height * Y) - (near_width * X);
		NTR = near_center + (near_height * Y) + (near_width * X);
		NBL = near_center - (near_height * Y) - (near_width * X);
		NBR = near_center - (near_height * Y) + (near_width * X);
		FTL = far_center  + (far_height  * Y) - (far_width  * X);
		FTR = far_center  + (far_height  * Y) + (far_width  * X);
		FBL = far_center  - (far_height  * Y) - (far_width  * X);
		FBR = far_center  - (far_height  * Y) + (far_width  * X);

		//printf( "near_center=%.3f,%.3f,%.3f\n", near_center.x, near_center.y, near_center.z );
		//printf( "far_center=%.3f,%.3f,%.3f\n", far_center.x, far_center.y, far_center.z );
		//printf( "near_width=%.3f, near_height=%.3f, far_width=%.3f, far_height=%.3f\n", near_width, near_height, far_width, far_height );
        //printf( "NTR=%.3f,%.3f,%.3f NTL=%.3f,%.3f,%.3f, FTL=%.3f,%.3f,%.3f\n", NTR.x,NTR.y,NTR.z, NTL.x,NTL.y,NTL.z, FTL.x,FTL.y,FTL.z );

		frustum_planes[0].set3Points( NTR, NTL, FTL );	// TOP
		frustum_planes[1].set3Points( NBL, NBR, FBR );	// BOTTOM
		frustum_planes[2].set3Points( NTL, NBL, FBL );	// LEFT
		frustum_planes[3].set3Points( NBR, NTR, FBR );	// RIGHT

		if( vfc_type == GEOMETRIC )
		{
			frustum_planes[4].set3Points( NTL, NTR, NBR );	// NEAR
			frustum_planes[5].set3Points( FTR, FTL, FBL );	// FAR
		}
		else if( vfc_type == RADAR )
		{
			// TOP: /////////////////////////////////////////////////////////////////////
			vec3 aux, normal, point;
			vec3 n_h_Y = near_height * Y;
			vec3 n_w_X = near_width  * X;
			aux		= (near_center + n_h_Y) - CAM_POS;
			normal	= glm::cross( aux, X );
			point   = near_center + n_h_Y;
			frustum_planes[0].setNormalAndPoint( normal, point );
			// BOTTOM: /////////////////////////////////////////////////////////////////
			aux		= (near_center - n_h_Y) - CAM_POS;
			normal	= glm::cross( X, aux );
			point   = near_center - n_h_Y;
			frustum_planes[2].setNormalAndPoint( normal, point );
			// LEFT: ////////////////////////////////////////////////////////////////////
			aux		= (near_center - n_w_X ) - CAM_POS;
			normal	= glm::cross( aux, Y );
			point   = near_center - n_w_X;
			frustum_planes[2].setNormalAndPoint( normal, point );
			// RIGHT: ///////////////////////////////////////////////////////////////////
			aux		= (near_center + n_w_X ) - CAM_POS;
			normal	= glm::cross( Y, aux );
			point   = near_center + n_w_X;
			frustum_planes[3].setNormalAndPoint( normal, point );
			// NEAR: ////////////////////////////////////////////////////////////////////
			normal = -Z;
			frustum_planes[4].setNormalAndPoint( normal, near_center );
			// FAR: /////////////////////////////////////////////////////////////////////
			frustum_planes[5].setNormalAndPoint(  Z, far_center  );
		}
	}
}

//=======================================================================================

void Frustum::setFovY( float fov )
{
	FOVY		= fov;
	TANG		= tan( DEG2RAD * FOVY * 0.5f );
	near_height = near_distance * TANG;
	near_width  = near_height   * RATIO;
	far_height  = far_distance  * TANG;
	far_width   = far_height    * RATIO;
	if( vfc_type == RADAR )
	{
		sphereFactorY = 1.0f / cos( FOVY );
		sphereFactorX = 1.0f / cos( atan( TANG * RATIO ) );
	}
}

//=======================================================================================

float Frustum::getRatio( void )
{
	return RATIO;
}

//=======================================================================================

void Frustum::updateRatio( void )
{
	RATIO		= proj_man->getAspectRatio();
	near_height = near_distance * TANG;
	near_width  = near_height   * RATIO;
	far_height  = far_distance  * TANG;
	far_width   = far_height    * RATIO;
	if( vfc_type == RADAR )
	{
		sphereFactorY = 1.0f / cos( FOVY );
		sphereFactorX = 1.0f / cos( atan( TANG * RATIO ) );
	}
}

//=======================================================================================

void Frustum::setNearD( float nearD )
{
	near_distance = nearD;
	near_height   = near_distance * TANG;
	near_width    = near_height   * RATIO;
	far_height    = far_distance  * TANG;
	far_width     = far_height    * RATIO;
	if( vfc_type == RADAR )
	{
		sphereFactorY = 1.0f / cos( FOVY );
		sphereFactorX = 1.0f / cos( atan( TANG * RATIO ) );
	}
}

//=======================================================================================

void Frustum::setFarD( float farD )
{
	far_distance = farD;
	near_height  = near_distance * TANG;
	near_width   = near_height   * RATIO;
	far_height   = far_distance  * TANG;
	far_width    = far_height    * RATIO;
	if( vfc_type == RADAR )
	{
		sphereFactorY = 1.0f / cos( FOVY );
		sphereFactorX = 1.0f / cos( atan( TANG * RATIO ) );
	}
}

//=======================================================================================

void Frustum::drawPoints( void )
{
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT );
	{
		glDisable( GL_LIGHTING );
		glDisable( GL_TEXTURE_2D );
		glColor3f( fR, fG, fB );
		glPointSize( 4.0f );
		glBegin( GL_POINTS );
		{
			glVertex3f( NTL.x, NTL.y, NTL.z );
			glVertex3f( NTR.x, NTR.y, NTR.z );
			glVertex3f( NBL.x, NBL.y, NBL.z );
			glVertex3f( NBR.x, NBR.y, NBR.z );

			glVertex3f( FTL.x, FTL.y, FTL.z );
			glVertex3f( FTR.x, FTR.y, FTR.z );
			glVertex3f( FBL.x, FBL.y, FBL.z );
			glVertex3f( FBR.x, FBR.y, FBR.z );
		}
		glEnd();
		glPointSize( 1 );
	}
	glPopAttrib();
}

//=======================================================================================

void Frustum::drawLines( void )
{
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT );
	{
		glDisable( GL_LIGHTING );
		glDisable( GL_TEXTURE_2D );
		glColor3f( fR, fG, fB );
		glLineWidth( 1.0f );
		//near plane
		glBegin( GL_LINE_LOOP );
		{
			glVertex3f( NTL.x, NTL.y, NTL.z );
			glVertex3f( NTR.x, NTR.y, NTR.z );
			glVertex3f( NBR.x, NBR.y, NBR.z );
			glVertex3f( NBL.x, NBL.y, NBL.z );
		}
		glEnd();
		glBegin( GL_LINES );
			glVertex3f( (NTL.x + NTR.x) / 2.0f,
						(NTL.y + NTR.y) / 2.0f,
						(NTL.z + NTR.z) / 2.0f );
			glVertex3f( (NBL.x + NBR.x) / 2.0f,
						(NBL.y + NBR.y) / 2.0f,
						(NBL.z + NBR.z) / 2.0f );
			glVertex3f( (NTL.x + NBL.x) / 2.0f,
						(NTL.y + NBL.y) / 2.0f,
						(NTL.z + NBL.z) / 2.0f );
			glVertex3f( (NTR.x + NBR.x) / 2.0f,
						(NTR.y + NBR.y) / 2.0f,
						(NTR.z + NBR.z) / 2.0f );
		glEnd();
		//far plane
		glBegin( GL_LINE_LOOP );
		{
			glVertex3f( FTR.x, FTR.y, FTR.z );
			glVertex3f( FTL.x, FTL.y, FTL.z );
			glVertex3f( FBL.x, FBL.y, FBL.z );
			glVertex3f( FBR.x, FBR.y, FBR.z );
		}
		glEnd();
		glBegin( GL_LINES );
			glVertex3f( (FTL.x + FTR.x) / 2.0f,
						(FTL.y + FTR.y) / 2.0f,
						(FTL.z + FTR.z) / 2.0f );
			glVertex3f( (FBL.x + FBR.x) / 2.0f,
						(FBL.y + FBR.y) / 2.0f,
						(FBL.z + FBR.z) / 2.0f );
			glVertex3f( (FTL.x + FBL.x) / 2.0f,
						(FTL.y + FBL.y) / 2.0f,
						(FTL.z + FBL.z) / 2.0f );
			glVertex3f( (FTR.x + FBR.x) / 2.0f,
						(FTR.y + FBR.y) / 2.0f,
						(FTR.z + FBR.z) / 2.0f );
		glEnd();
		//bottom plane
		glBegin( GL_LINE_LOOP );
		{
			glVertex3f( NBL.x, NBL.y, NBL.z );
			glVertex3f( NBR.x, NBR.y, NBR.z );
			glVertex3f( FBR.x, FBR.y, FBR.z );
			glVertex3f( FBL.x, FBL.y, FBL.z );
		}
		glEnd();
		glBegin( GL_LINES );
			glVertex3f( (NBL.x + NBR.x) / 2.0f,
						(NBL.y + NBR.y) / 2.0f,
						(NBL.z + NBR.z) / 2.0f );
			glVertex3f( (FBL.x + FBR.x) / 2.0f,
						(FBL.y + FBR.y) / 2.0f,
						(FBL.z + FBR.z) / 2.0f );
			glVertex3f( (NBL.x + FBL.x) / 2.0f,
						(NBL.y + FBL.y) / 2.0f,
						(NBL.z + FBL.z) / 2.0f );
			glVertex3f( (NBR.x + FBR.x) / 2.0f,
						(NBR.y + FBR.y) / 2.0f,
						(NBR.z + FBR.z) / 2.0f );
		glEnd();
		//top plane
		glBegin( GL_LINE_LOOP );
		{
			glVertex3f( NTR.x, NTR.y, NTR.z );
			glVertex3f( NTL.x, NTL.y, NTL.z );
			glVertex3f( FTL.x, FTL.y, FTL.z );
			glVertex3f( FTR.x, FTR.y, FTR.z );
		}
		glEnd();
		glBegin( GL_LINES );
			glVertex3f( (NTL.x + NTR.x) / 2.0f,
						(NTL.y + NTR.y) / 2.0f,
						(NTL.z + NTR.z) / 2.0f );
			glVertex3f( (FTL.x + FTR.x) / 2.0f,
						(FTL.y + FTR.y) / 2.0f,
						(FTL.z + FTR.z) / 2.0f );
			glVertex3f( (NTL.x + FTL.x) / 2.0f,
						(NTL.y + FTL.y) / 2.0f,
						(NTL.z + FTL.z) / 2.0f );
			glVertex3f( (NTR.x + FTR.x) / 2.0f,
						(NTR.y + FTR.y) / 2.0f,
						(NTR.z + FTR.z) / 2.0f );
		glEnd();
		//left plane
		glBegin( GL_LINE_LOOP );
		{
			glVertex3f( NTL.x, NTL.y, NTL.z );
			glVertex3f( NBL.x, NBL.y, NBL.z );
			glVertex3f( FBL.x, FBL.y, FBL.z );
			glVertex3f( FTL.x, FTL.y, FTL.z );
		}
		glEnd();
		glBegin( GL_LINES );
			glVertex3f( (NBL.x + NTL.x) / 2.0f,
						(NBL.y + NTL.y) / 2.0f,
						(NBL.z + NTL.z) / 2.0f );
			glVertex3f( (FBL.x + FTL.x) / 2.0f,
						(FBL.y + FTL.y) / 2.0f,
						(FBL.z + FTL.z) / 2.0f );
			glVertex3f( (NTL.x + FTL.x) / 2.0f,
						(NTL.y + FTL.y) / 2.0f,
						(NTL.z + FTL.z) / 2.0f );
			glVertex3f( (NBL.x + FBL.x) / 2.0f,
						(NBL.y + FBL.y) / 2.0f,
						(NBL.z + FBL.z) / 2.0f );
		glEnd();
		// right plane
		glBegin( GL_LINE_LOOP );
		{
			glVertex3f( NBR.x, NBR.y, NBR.z );
			glVertex3f( NTR.x, NTR.y, NTR.z );
			glVertex3f( FTR.x, FTR.y, FTR.z );
			glVertex3f( FBR.x, FBR.y, FBR.z );
		}
		glEnd();
		glBegin( GL_LINES );
			glVertex3f( (NBR.x + NTR.x) / 2.0f,
						(NBR.y + NTR.y) / 2.0f,
						(NBR.z + NTR.z) / 2.0f );
			glVertex3f( (FBR.x + FTR.x) / 2.0f,
						(FBR.y + FTR.y) / 2.0f,
						(FBR.z + FTR.z) / 2.0f );
			glVertex3f( (NTR.x + FTR.x) / 2.0f,
						(NTR.y + FTR.y) / 2.0f,
						(NTR.z + FTR.z) / 2.0f );
			glVertex3f( (NBR.x + FBR.x) / 2.0f,
						(NBR.y + FBR.y) / 2.0f,
						(NBR.z + FBR.z) / 2.0f );
		glEnd();
		glLineWidth( 1 );
	}
	glPopAttrib();
}

//=======================================================================================

void Frustum::drawPlanes( void )
{
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT );
	{
		glDisable( GL_LIGHTING );
		glDisable( GL_TEXTURE_2D );
		glColor3f( fR, fG, fB );
		glBegin( GL_QUADS );
		{
			//near plane
			glVertex3f( NTL.x, NTL.y, NTL.z );
			glVertex3f( NTR.x, NTR.y, NTR.z );
			glVertex3f( NBR.x, NBR.y, NBR.z );
			glVertex3f( NBL.x, NBL.y, NBL.z );
			//far plane
			glVertex3f( FTR.x, FTR.y, FTR.z );
			glVertex3f( FTL.x, FTL.y, FTL.z );
			glVertex3f( FBL.x, FBL.y, FBL.z );
			glVertex3f( FBR.x, FBR.y, FBR.z );
			//bottom plane
			glVertex3f( NBL.x, NBL.y, NBL.z );
			glVertex3f( NBR.x, NBR.y, NBR.z );
			glVertex3f( FBR.x, FBR.y, FBR.z );
			glVertex3f( FBL.x, FBL.y, FBL.z );
			//top plane
			glVertex3f( NTR.x, NTR.y, NTR.z );
			glVertex3f( NTL.x, NTL.y, NTL.z );
			glVertex3f( FTL.x, FTL.y, FTL.z );
			glVertex3f( FTR.x, FTR.y, FTR.z );
			//left plane
			glVertex3f( NTL.x, NTL.y, NTL.z );
			glVertex3f( NBL.x, NBL.y, NBL.z );
			glVertex3f( FBL.x, FBL.y, FBL.z );
			glVertex3f( FTL.x, FTL.y, FTL.z );
			// right plane
			glVertex3f( NBR.x, NBR.y, NBR.z );
			glVertex3f( NTR.x, NTR.y, NTR.z );
			glVertex3f( FTR.x, FTR.y, FTR.z );
			glVertex3f( FBR.x, FBR.y, FBR.z );
		}
		glEnd();
	}
	glPopAttrib();
}

//=======================================================================================

void Frustum::drawNormals( void )
{
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT );
	{
		glDisable( GL_LIGHTING );
		glDisable( GL_TEXTURE_2D );
		glColor3f( fR, fG, fB );
		vec3 a, b;
		glLineWidth( 1.0f );
		glBegin( GL_LINES );
		{
			// near
			a = 0.25f * (NTR + NTL + NBR + NBL);
			b = a + 5.0f * frustum_planes[NEARP].N;
			glVertex3f( a.x, a.y, a.z );
			glVertex3f( b.x, b.y, b.z );
			// far
			a = 0.25f * (FTR + FTL + FBR + FBL);
			b = a + 5.0f * frustum_planes[FARP].N;
			glVertex3f( a.x, a.y, a.z );
			glVertex3f( b.x, b.y, b.z );
			// left
			a = 0.25f * (FTL + FBL + NBL + NTL);
			b = a + 5.0f * frustum_planes[LEFT].N;
			glVertex3f( a.x, a.y, a.z );
			glVertex3f( b.x, b.y, b.z );
			// right
			a = 0.25f * (FTR + NBR + FBR + NTR);
			b = a + 5.0f * frustum_planes[RIGHT].N;
			glVertex3f( a.x, a.y, a.z );
			glVertex3f( b.x, b.y, b.z );
			// top
			a = 0.25f * (FTR + FTL + NTR + NTL);
			b = a + 5.0f * frustum_planes[TOP].N;
			glVertex3f( a.x, a.y, a.z );
			glVertex3f( b.x, b.y, b.z );
			// bottom
			a = 0.25f * (FBR + FBL + NBR + NBL);
			b = a + 5.0f * frustum_planes[BOTTOM].N;
			glVertex3f( a.x, a.y, a.z );
			glVertex3f( b.x, b.y, b.z );
		}
		glEnd();
		glLineWidth( 1 );
	}
	glPopAttrib();
}

//=======================================================================================

unsigned int Frustum::getCulledCount( void )
{
	return culled_count;
}

//=======================================================================================

void Frustum::incCulledCount( void )
{
	culled_count++;
}

//=======================================================================================

void Frustum::resetCulledCount( void )
{
	culled_count = 0;
}

//=======================================================================================

float Frustum::getNearD( void )
{
	return near_distance;
}

//=======================================================================================

float Frustum::getFarD( void )
{
	return far_distance;
}

//=======================================================================================

float Frustum::getFovY( void )
{
	return FOVY;
}

//=======================================================================================

float Frustum::getNearW( void )
{
	return near_width;
}

//=======================================================================================

float Frustum::getNearH( void )
{
	return near_height;
}

//=======================================================================================

float Frustum::getFarW( void )
{
	return far_width;
}

//=======================================================================================

float Frustum::getFarH( void )
{
	return far_height;
}

//=======================================================================================

vec3& Frustum::getX( void )
{
	return X;
}

//=======================================================================================

vec3& Frustum::getY( void )
{
	return Y;
}

//=======================================================================================

vec3& Frustum::getZ( void )
{
	return Z;
}

//=======================================================================================

