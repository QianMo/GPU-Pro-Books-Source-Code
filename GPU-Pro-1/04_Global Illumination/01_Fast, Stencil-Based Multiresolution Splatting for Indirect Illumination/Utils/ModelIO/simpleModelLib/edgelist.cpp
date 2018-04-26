#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mesh.h"
#include "funcs.h"

void  EdgeListConstruct(Solid ** solid )
{
   Face * fhead, *tf;
   HalfEdge *the, *he_mate;

   fhead = (*solid)->sfaces;
   assert( fhead);
   tf = fhead;
   do{
   
   the = tf->floop->ledges;
   do{

     if( the->hedge == NIL ){
     
       he_mate = HalfEdgeMate(the);
       EdgeConstruct( solid, the, he_mate);
     }

   the = the->next;

   }while( the != tf->floop->ledges );   


   tf = tf->next;
   }while( tf != fhead );

}

// This is a naive search, and might take a while, be sure to
//  Give the option to print out some status.
void  EdgeListConstructVerbose(Solid ** solid, int numFaces )
{
   Face * fhead, *tf;
   HalfEdge *the, *he_mate;
   int i = 0;

   printf("    (-) Constructing edge list (%8.4f%% done)...", 0.0f ); fflush( stdout );

   fhead = (*solid)->sfaces;
   assert( fhead);
   tf = fhead;
   do{
   
   the = tf->floop->ledges;
   do{

     if( the->hedge == NIL ){
     
       he_mate = HalfEdgeMate(the);
       EdgeConstruct( solid, the, he_mate);
     }

   the = the->next;

   }while( the != tf->floop->ledges );   


   if (i%100 == 0)
   {
		printf("\r    (-) Constructing edge list (%8.4f%% done)...", 100.0f*((float)i/numFaces) ); 
		fflush( stdout );
   }

   i++;
   tf = tf->next;
   }while( tf != fhead );

	
   printf("\r    (-) Constructing edge list (%8.4f%% done)...\n", 100.0f ); fflush( stdout );

}


void  EdgeListDestruct(Solid ** solid )
{
	 Edge * te;


	 while( (*solid)->sedges ){
	 te = (*solid)->sedges;
	 assert( te->he1 == NIL );
	 assert( te->he2 == NIL );
	 EdgeDestruct(&te);
	 }

}



