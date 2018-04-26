#include <stdio.h>
#include <stdlib.h>
#define  VERTEX    101
#define  HALFEDGE  102
#define  EDGE      103
#define  LOOP      104
#define  FACE      105
#define  SOLID     106

#define  NIL 0
#define  TRUE 1
#define  FALSE 0

#define SWAP(t,x,y)     { t = x; x = y; y = t; }


#define NEW(p,type)     if ((p=(type *) malloc (sizeof(type))) == NIL) {\
                                printf ("Out of Memory!\n");\
                                exit(0);\
                        }

#define FREE(p)         if (p) { free ((char *) p); p = NIL; }


#define ADD( head, p )  if ( head )  { \
                                p->prev = head->prev; \
                                p->next = head; \
                                head->prev = p; \
                                p->prev->next = p; \
                        } \
                        else { \
                                head = p; \
                                head->next = head->prev = p; \
                        }

#define DELETE( head, p )   if ( head )  { \
                                if ( head == head->next ) \
                                        head = NIL;  \
                                else if ( p == head ) \
                                        head = head->next; \
                                p->next->prev = p->prev;  \
                                p->prev->next = p->next;  \
                                FREE( p ); \
                        } 





typedef int                  Id;
typedef struct solid      Solid;
typedef struct face       Face;
typedef struct loop       Loop;
typedef struct halfedge   HalfEdge;
typedef struct vertex     Vertex;
typedef struct edge       Edge;
typedef struct node       Node;

struct node{

 int    type;
 void * p;
 double v;
 
 Node * next;
 Node * prev;
};



struct solid{

   Face    *sfaces;
   Edge    *sedges;
   Vertex  *sverts;
   double   center[3];
};

struct face{
  
   Id          faceno;

   Loop      * floop;
   Solid     * fsolid;
   double      normal[3];

   Face      * next;
   Face      * prev;

   int       alivef;
};

struct edge{

	Id edgeno;

	HalfEdge  *he1;
	HalfEdge  *he2;
	Solid     *esolid;

	Edge      *next;
	Edge      *prev;

	int      alive;

};

struct halfedge{


	Edge     *hedge;
	Loop     *hloop;
	Vertex   *hvert;

	HalfEdge *next;
	HalfEdge *prev;

	int     aliveh;

};


struct vertex{

	Id        vertexno;
	HalfEdge  *vedge;
        double    gauss_cur;
	double    vcoord[3];
        double    ncoord[3];

	Vertex    *next;
	Vertex    *prev;

	int      alivev;
};



struct loop{
  
       HalfEdge *ledges;
       Face     *lface;

       int      alivel;
};













