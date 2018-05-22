
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _KDTREE_H_
#define _KDTREE_H_

#include <stdlib.h>
//#ifndef NULL
//#define NULL 0
//#endif

template <class TYPE>
struct KdNode {
	KdNode <TYPE> *lower, *upper;
	unsigned int index;
	TYPE *point;
};

/*
struct KdMemBlock {
	unsigned char *mem;
	KdMemBlock *next;
};
*/

template <class TYPE>
class KdTree {
public:
	KdTree(const unsigned int nComp, const unsigned int capasity){
		top = NULL;
		nComponents = nComp;
		count = 0;

		curr = mem = (unsigned char *) malloc(capasity * (sizeof(KdNode <TYPE>) + nComp * sizeof(TYPE)));
/*
		memBlock = new KdMemBlock;
		memBlock->mem = NULL;
		memBlock->next = NULL;
		memIndex = 1048576;
*/
	}

	~KdTree(){
/*
		while (memBlock){
			delete memBlock->mem;
			KdMemBlock *next = memBlock->next;
			delete memBlock;
			memBlock = next;
		}
*/
		free(mem);

		//freeNode(top);
	}

	unsigned int addUnique(const TYPE *point){
		if (top != NULL){
			return addUniqueToNode(top, point);
		} else {
			top = newNode(point);
			return 0;
		}
	}

	void clear(){
		curr = mem;
		//freeNode(top);

		top = NULL;
		count = 0;
	}

	const unsigned int getCount() const { return count; }

private:
	unsigned int addUniqueToNode(KdNode <TYPE> *node, const TYPE *point){
		unsigned int comp = 0;

		while (true){
			if (point[comp] < node->point[comp]){
				if (node->lower){
					node = node->lower;
				} else {
					node->lower = newNode(point);
					return node->lower->index;
				}
			} else if (point[comp] > node->point[comp]){
				if (node->upper){
					node = node->upper;
				} else {
					node->upper = newNode(point);
					return node->upper->index;
				}
			} else if (isEqualToNode(node, point)){
				return node->index;
			} else {
				if (node->upper){
					node = node->upper;
				} else {
					node->upper = newNode(point);
					return node->upper->index;
				}
			}
			if (++comp == nComponents) comp = 0;
		}


		/*if (isEqualToNode(node, point)){
			return node->index;
		} else {
			if (point[comp] < node->point[comp]){
				if (node->lower){
					if (++comp == nComponents) comp = 0;
					return addUniqueToNode(node->lower, point, comp);
				} else {
					node->lower = newNode(point);
					return node->lower->index;
				}
			} else {
				if (node->upper){
					if (++comp == nComponents) comp = 0;
					return addUniqueToNode(node->upper, point, comp);
				} else {
					node->upper = newNode(point);
					return node->upper->index;
				}
			}
		}*/
	}

	KdNode <TYPE> *newNode(const TYPE *point){
		//KdNode <TYPE> *node = new KdNode<TYPE>;
		//node->point = new TYPE[nComponents];

		KdNode <TYPE> *node = (KdNode <TYPE> *) newMem(sizeof(KdNode <TYPE>));
		node->point = (TYPE *) newMem(nComponents * sizeof(TYPE));


		//for (unsigned int i = 0; i < nComponents; i++) node->point[i] = point[i];
		memcpy(node->point, point, nComponents * sizeof(TYPE));

		node->lower = NULL;
		node->upper = NULL;
		node->index = count++;
		return node;
	}

	/*void freeNode(KdNode <TYPE> *node){
		if (node != NULL){
			delete node->point;
			freeNode(node->lower);
			freeNode(node->upper);
			delete node;
		}
	}*/

	bool isEqualToNode(const KdNode <TYPE> *node, const TYPE *point){
		unsigned int i = 0;
		do {
			//if (fabsf(node->point[i] - point[i]) > 0.001f) return false;
			if (node->point[i] != point[i]) return false;
			i++;
		} while (i < nComponents);
		return true;
	}

	void *newMem(const unsigned int size){
/*
		unsigned char *rmem = memBlock->mem + memIndex;
		memIndex += size;
		if (memIndex > 1048576){
			KdMemBlock *newMemBlock = new KdMemBlock;
			newMemBlock->next = memBlock;
			memBlock = newMemBlock;
			rmem = memBlock->mem = (unsigned char *) malloc(1048576);
			memIndex = size;
		}
		return rmem;
*/

		unsigned char *rmem = curr;
		curr += size;
		return rmem;
	}

	KdNode <TYPE> *top;
	unsigned int nComponents;
	unsigned int count;

	unsigned char *mem, *curr;
//	KdMemBlock *memBlock;
//	unsigned int memIndex;
};

#endif // _KDTREE_H_
