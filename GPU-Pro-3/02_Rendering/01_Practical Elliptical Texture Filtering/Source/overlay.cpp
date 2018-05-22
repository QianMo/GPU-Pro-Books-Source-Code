/**
*	EWA filtering on the GPU
*	Copyright 2010-2011 Pavlos Mavridis, All rights reserved
*/

#include "overlay.h"

Overlay_s::Overlay_s(float w,float h):width(w),height(h){
//	BuildFont();
};

void Overlay_s::Begin(void){
		
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0,width,height,0,-10.0f,10.0f);	//to (0,0) panw aristera
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_TEXTURE_2D);
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
}

void Overlay_s::End(void){
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	
	glDisable( GL_BLEND );
}

//I think the credit for this function goes to NEHE
void Overlay_s::BuildFont(void){

	if(!font.LoadTGA("Textures\\Font.tga",GL_LINEAR,GL_LINEAR,0)){
			printf("FATAL ERROR: Loading Font");
			exit(EXIT_FAILURE);
	}
		base=glGenLists(256);
		font.Bind();
		for (int loop1=0; loop1<256; loop1++)					// Loop Through All 256 Lists
		{
			float cx=float(loop1%16)/16.0f;						// X Position Of Current Character
			float cy=float(loop1/16)/16.0f;						// Y Position Of Current Character

			glNewList(base+loop1,GL_COMPILE);
				
				glBegin(GL_QUADS);								// Use A Quad For Each Character
					glTexCoord2f(cx,1.0f-cy-0.0625f);			// Texture Coord (Bottom Left)
					glVertex2d(0,16);							// Vertex Coord (Bottom Left)
					glTexCoord2f(cx+0.0625f,1.0f-cy-0.0625f);	// Texture Coord (Bottom Right)
					glVertex2i(16,16);							// Vertex Coord (Bottom Right)
					glTexCoord2f(cx+0.0625f,1.0f-cy-0.001f);	// Texture Coord (Top Right)
					glVertex2i(16,0);							// Vertex Coord (Top Right)
					glTexCoord2f(cx,1.0f-cy-0.001f);			// Texture Coord (Top Left)
					glVertex2i(0,0);							// Vertex Coord (Top Left)
				glEnd();										// Done Building Our Quad (Character)
				glTranslated(9,0,0);							// Move To The Right Of The Character
			glEndList();										// Done Building The Display List
		}														// Loop Until All 256 Are Built
}

GLvoid Overlay_s::KillFont(GLvoid){
		glDeleteLists(base,256);								
}

GLvoid Overlay_s::Print(GLfloat x, GLfloat y, int set, const char *txt,float alpha){
		if (!txt)										
			return;												

		font.Bind();
		glLoadIdentity();

		int tmp=strlen(txt);
		glListBase(base-32+(128*set));	

		//render text shadow
		glPushMatrix();
			glTranslatef(x+1.5,y+1.5,-0.5);
			glScalef(1.2,1.5,1);
			glColor4f(0.01,0.01,0.01,alpha);
			glCallLists(tmp,GL_UNSIGNED_BYTE, txt);	
		glPopMatrix();

		glTranslatef(x,y,-0.5);
		glScalef(1.2,1.5,1);
		glColor4f(1,0.02,0.02,alpha);
		glCallLists(tmp,GL_UNSIGNED_BYTE, txt);	

		glColor4f(1,1,1,1);
	}
