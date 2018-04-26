/******************************************************************/
/* Quad.cpp                                                       */
/* -----------------------                                        */
/*                                                                */
/* The file defines a simple drawable OpenGL quad type.           */
/*                                                                */
/* Chris Wyman (01/01/2008)                                       */
/******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "Utils/GLee.h"
#include <GL/glut.h>
#include "Quad.h"
#include "Scene/Scene.h"
#include "Utils/ImageIO/imageIO.h"


void Quad::Draw( Scene *s, unsigned int matlFlags, unsigned int optionFlags )
{
	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		matl->Enable( s, matlFlags );
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		matl->EnableOnlyTextures( s );

	glBegin( GL_QUADS );
		glTexCoord3f( tex0.X(), tex0.Y(), tex0.Z() );
		glNormal3fv( norm0.GetDataPtr() );
		glVertex3fv( vert0.GetDataPtr() );
		glTexCoord3f( tex1.X(), tex1.Y(), tex1.Z() );
		glNormal3fv( norm1.GetDataPtr() );
		glVertex3fv( vert1.GetDataPtr() );
		glTexCoord3f( tex2.X(), tex2.Y(), tex2.Z() );
		glNormal3fv( norm2.GetDataPtr() );
		glVertex3fv( vert2.GetDataPtr() );
		glTexCoord3f( tex3.X(), tex3.Y(), tex3.Z() );
		glNormal3fv( norm3.GetDataPtr() );
		glVertex3fv( vert3.GetDataPtr() );
	glEnd();

	if (!(matlFlags & MATL_CONST_AVOIDMATERIAL) && matl)
		matl->Disable();
	else if ((matlFlags & MATL_FLAGS_ENABLEONLYTEXTURES) && matl)
		matl->DisableOnlyTextures();
}

// Draw this object (or it's sub-objects only if they have some property)
void Quad::DrawOnly( Scene *s, unsigned int propertyFlags, unsigned int matlFlags, unsigned int optionFlags )
{
	if ((propertyFlags & flags) == propertyFlags) 
		this->Draw( s, matlFlags, optionFlags );
}



Quad::Quad( FILE *f, Scene *s ) :
	Object( s->GetDefaultMaterial() ), 
	vert0(0,0,0), vert1(0.5f,0,0), vert2(0.5f,0.5f,0), vert3(0,0.5f,0),
	tex0(0,0,0), tex1(0.5f,0,0), tex2(0.5f,0.5f,0), tex3(0,0.5f,0)
{
	bool normalsDefined=false;
	bool useEdges=false, useTexDelta=false;
	Vector e1, e2, td1, td2;

	// Search the scene file.
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;
		if (TestCommonObjectProperties( token, ptr, s, f ))
			continue;
		else if (!strcmp(token,"p0") || !strcmp(token,"v0") ||
			!strcmp(token,"pt0") || !strcmp(token,"vertex0") || !strcmp(token,"point0"))
			vert0 = Point( ptr );
		else if (!strcmp(token,"p1") || !strcmp(token,"v1") ||
			!strcmp(token,"pt1") || !strcmp(token,"vertex1") || !strcmp(token,"point1"))
			vert1 = Point( ptr );
		else if (!strcmp(token,"p2") || !strcmp(token,"v2") ||
			!strcmp(token,"pt2") || !strcmp(token,"vertex2") || !strcmp(token,"point2"))
			vert2 = Point( ptr );
		else if (!strcmp(token,"p3") || !strcmp(token,"v3") ||
			!strcmp(token,"pt3") || !strcmp(token,"vertex3") || !strcmp(token,"point3"))
			vert3 = Point( ptr );
		else if (!strcmp(token,"t0") || !strcmp(token,"tex0") || !strcmp(token,"texture0") || !strcmp(token,"texat0"))
			tex0 = Vector( ptr );
		else if (!strcmp(token,"t1") || !strcmp(token,"tex1") || !strcmp(token,"texture1") || !strcmp(token,"texat1"))
			tex1 = Vector( ptr );
		else if (!strcmp(token,"t2") || !strcmp(token,"tex2") || !strcmp(token,"texture2") || !strcmp(token,"texat2"))
			tex2 = Vector( ptr );
		else if (!strcmp(token,"t3") || !strcmp(token,"tex3") || !strcmp(token,"texture3") || !strcmp(token,"texat3"))
			tex3 = Vector( ptr );
		else if (!strcmp(token,"e1") || !strcmp(token,"edge1"))
		{
			e1 = Vector( ptr ); useEdges = true;
		}
		else if (!strcmp(token,"e2") || !strcmp(token,"edge2"))
		{
			e2 = Vector( ptr ); useEdges = true;
		}
		else if (!strcmp(token,"texdelta1") || !strcmp(token,"tedge1") || !strcmp(token,"tdel1"))
		{
			td1 = Vector( ptr ); useTexDelta = true;
		}
		else if (!strcmp(token,"texdelta2") || !strcmp(token,"tedge2") || !strcmp(token,"tdel2"))
		{
			td2 = Vector( ptr ); useTexDelta = true;
		}
		else if (!strcmp(token,"n0") || !strcmp(token,"norm0") || !strcmp(token,"normal0"))
		{
			norm0 = Vector( ptr );
			norm0.Normalize();
			normalsDefined = true;
		}
		else if (!strcmp(token,"n1") || !strcmp(token,"norm1") || !strcmp(token,"normal1"))
		{
			norm1 = Vector( ptr );
			norm1.Normalize();
			normalsDefined = true;
		}
		else if (!strcmp(token,"n2") || !strcmp(token,"norm2") || !strcmp(token,"normal2"))
		{
			norm2 = Vector( ptr );
			norm2.Normalize();
			normalsDefined = true;
		}
		else if (!strcmp(token,"n3") || !strcmp(token,"norm3") || !strcmp(token,"normal3"))
		{
			norm3 = Vector( ptr );
			norm3.Normalize();
			normalsDefined = true;
		}
		else if (!strcmp(token,"height") || !strcmp(token,"cycles") || !strcmp(token,"mult") || !strcmp(token,"multiplier"))
			Warning("Experimental '%' keyword not supported in OpenGL Quads!", token);
		else
			Error("Unknown command '%s' when loading Quad!", token);
	}

	if (useEdges)
	{
		vert1 = vert0 + e1;
		vert2 = vert0 + e1 + e2;
		vert3 = vert0 + e2;
		if (!normalsDefined)
		{
			normalsDefined = true;
			norm0 = e1.Cross(e2);
			norm0.Normalize();
			norm1 = norm0;
			norm2 = norm0;
			norm3 = norm0;
		}
	}
	
	if (useTexDelta)
	{
		tex1 = tex0 + td1;
		tex2 = tex0 + td1 + td2;
		tex3 = tex0 + td2;
	}

	if (!normalsDefined)
	{
		e1 = vert1-vert0;
		e2 = vert3-vert0;
		norm0 = e1.Cross(e2);
		norm0.Normalize();
		norm1 = norm0;
		norm2 = norm0;
		norm3 = norm0;
	}

}

