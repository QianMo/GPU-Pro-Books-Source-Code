/******************************************************************/
/* Object.cpp                                                     */
/* -----------------------                                        */
/*                                                                */
/* The file defines a base object class that is inherited by more */
/*     complex object types.                                      */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#include "Object.h"
#include "Scene/Scene.h"


bool Object::TestCommonObjectProperties( char *keyword, char *restOfLine, Scene *s, FILE *f )
{
	char *ptr=restOfLine, token[256];
	
	if (!strcmp(keyword,"background") || !strcmp(keyword,"isbackground") || !strcmp(keyword,"bg"))
	{	
		flags |= OBJECT_FLAGS_ISBACKGROUND;
		return true;
	}
	else if (!strcmp(keyword,"foreground") || !strcmp(keyword,"isforeground"))
	{	
		flags |= OBJECT_FLAGS_ISFOREGROUND;
		return true;
	}
	else if (!strcmp(keyword,"castshadow") || !strcmp(keyword,"shadows") || !strcmp(keyword,"castshadows"))
	{	
		flags |= OBJECT_FLAGS_CASTSSHADOWS;
		return true;
	}
	else if (!strcmp(keyword,"reflective") || !strcmp(keyword,"isreflective") || !strcmp(keyword,"reflects"))
	{	
		flags |= OBJECT_FLAGS_ISREFLECTIVE;
		return true;
	}
	else if (!strcmp(keyword,"refractive") || !strcmp(keyword,"isrefractive") || !strcmp(keyword,"refracts"))
	{	
		flags |= OBJECT_FLAGS_ISREFRACTIVE;
		return true;
	}
	else if (!strcmp(keyword,"flag1") || !strcmp(keyword,"userflag1") || !strcmp(keyword,"type1"))
	{	
		flags |= OBJECT_USER_FLAG1;
		return true;
	}
	else if (!strcmp(keyword,"flag2") || !strcmp(keyword,"userflag2") || !strcmp(keyword,"type2"))
	{	
		flags |= OBJECT_USER_FLAG2;
		return true;
	}
	else if (!strcmp(keyword,"material"))
	{
		StripLeadingTokenToBuffer( ptr, token );
		Material *m = s->ExistingMaterialFromFile( token );
		if (!m)	
		{
			m = s->LoadMaterial( ptr, f );
			s->AddMaterial( m );
		}
		SetMaterial( m );
		return true;
	}
	else if (!strcmp(keyword,"move") || !strcmp(keyword,"movement"))
	{
		objMove = new ObjectMovement( f, s );
		return true;
	}
		
	return false;
}


