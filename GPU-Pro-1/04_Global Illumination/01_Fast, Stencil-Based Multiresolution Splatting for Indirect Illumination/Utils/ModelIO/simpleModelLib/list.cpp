#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mesh.h"
#include "funcs.h"


Node *NodeNew(){
  Node * n;

  NEW(n,Node);
  n->p = NIL;

  return n;
}

void NodeConstruct(Node ** root, void * pointer, int type ){

  Node * node;
  node = NodeNew( );
  node->p = pointer;
  node->type = type;
  ADD( (*root), node);
 
}

void NodeDelete( Node ** list, Node ** node ){


	DELETE( (*list) ,(*node));


}


int ListInsertNode( Node ** root, void * pointer, int type ){

  Node * node = (*root);

  if( node )
    do{
      if( node->p == pointer ) return 0;
      node = node->next;
    }while( node != (*root) );

  NodeConstruct(root, pointer,type);
  return 1;
}




int ListDeleteNode( Node ** root, void * pointer, int type ){

  Node * node = (*root);

  if( node )
    do{
      if( node->p == pointer && node->type == type ){
	NodeDelete( root, &node );
	return 1;
      }
      node = node->next;
    }while( node != (*root) );

  return 0;
}


void  ListDestruct(Node ** root  )
{
	 Node * tn;


	 while( *root ){
	 tn = (*root);
	 NodeDelete(root,&tn);
	 }

}
/*
void main(){

 int a = 1;
 int b = 2;
 int c = 3;
 int d = 4;

 Node * root = NIL;
 Node * node;
 
 ListInsertNode( &root, (void*)&a, 0 ); 
 ListInsertNode( &root, (void*)&a, 0 ); 
 ListInsertNode( &root, (void*)&b, 0 ); 
 ListInsertNode( &root, (void*)&c, 0 ); 
 ListInsertNode( &root, (void*)&c, 0 ); 
 ListInsertNode( &root, (void*)&d, 0 ); 
 ListInsertNode( &root, (void*)&d, 0 ); 
 ListInsertNode( &root, (void*)&d, 0 ); 

 node = root;
 do{
   printf("%d\n",*((int *)node->p));
 node = node->next;
 }while( node != root );
 ListDeleteNode( &root, (void*)&c, 0 ); 

 node = root;
 do{
   printf("%d\n",*((int *)node->p));
 node = node->next;
 }while( node != root );
 ListDestruct( &root );
}

*/

