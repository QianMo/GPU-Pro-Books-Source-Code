
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

#ifndef _QUEUE_H_
#define _QUEUE_H_

template <class TYPE>
struct QueueNode {
	QueueNode <TYPE> *prev;
	QueueNode <TYPE> *next;
	TYPE object;
};

template <class TYPE>
class Queue {
public:
	Queue(){
		count = 0;
		first = NULL;
		last  = NULL;
		curr  = NULL;
		del   = NULL;
	}

	~Queue(){
		clear();
	}

	unsigned int getCount() const { return count; }

	void addFirst(const TYPE object){
		QueueNode <TYPE> *node = new QueueNode <TYPE>;
		node->object = object;
		insertNodeFirst(node);
		count++;
	}

	void addLast(const TYPE object){
		QueueNode <TYPE> *node = new QueueNode <TYPE>;
		node->object = object;
		insertNodeLast(node);
		count++;
	}

	void insertBeforeCurrent(const TYPE object){
		QueueNode <TYPE> *node = new QueueNode <TYPE>;
		node->object = object;
		insertNodeBefore(curr, node);
		count++;
	}

	void insertAfterCurrent(const TYPE object){
		QueueNode <TYPE> *node = new QueueNode <TYPE>;
		node->object = object;
		insertNodeAfter(curr, node);
		count++;
	}

	bool removeCurrent(){
		if (curr != NULL){
			releaseNode(curr);
			if (del) delete del;
			del = curr;
			count--;
		}
		return (curr != NULL);
	}

	bool goToFirst(){ return (curr = first) != NULL; }
	bool goToLast (){ return (curr = last ) != NULL; }
	bool goToPrev (){ return (curr = curr->prev) != NULL; }
	bool goToNext (){ return (curr = curr->next) != NULL; }
	bool goToObject(const TYPE object){
		curr = first;
		while (curr != NULL){
			if (object == curr->object) return true;
			curr = curr->next;
		}
		return false;
	}

	TYPE getCurrent() const { return curr->object; }
	void setCurrent(const TYPE object){ curr->object = object; }

	TYPE getPrev() const { return curr->prev->object; }
	TYPE getNext() const { return curr->next->object; }
	TYPE getPrevWrap() const { return ((curr->prev != NULL)? curr->prev : last)->object; }
	TYPE getNextWrap() const { return ((curr->next != NULL)? curr->next : first)->object; }

	void clear(){
		delete del;
		del = NULL;
		while (first){
			curr = first;
			first = first->next;
			delete curr;
		}
		last = curr = NULL;
		count = 0;
	}

	void moveCurrentToTop(){
		if (curr != NULL){
			releaseNode(curr);
			insertNodeFirst(curr);
		}
	}

protected:
	void insertNodeFirst(QueueNode <TYPE> *node){
		if (first != NULL){
			first->prev = node;
		} else {
			last = node;
		}
		node->next = first;
		node->prev = NULL;

		first = node;
	}

	void insertNodeLast(QueueNode <TYPE> *node){
		if (last != NULL){
			last->next = node;
		} else {
			first = node;
		}
		node->prev = last;
		node->next = NULL;

		last = node;
	}

	void insertNodeBefore(QueueNode <TYPE> *at, QueueNode <TYPE> *node){
		QueueNode <TYPE> *prev = at->prev;
		at->prev = node;
		if (prev){
			prev->next = node;
		} else {
			first = node;
		}
		node->next = at;
		node->prev = prev;
	}

	void insertNodeAfter(QueueNode <TYPE> *at, QueueNode <TYPE> *node){
		QueueNode <TYPE> *next = at->next;
		at->next = node;
		if (next){
			next->prev = node;
		} else {
			last = node;
		}
		node->prev = at;
		node->next = next;
	}

	void releaseNode(const QueueNode <TYPE> *node){
		if (node->prev == NULL){
			first = node->next;
		} else {
			node->prev->next = node->next;
		}
		if (node->next == NULL){
			last = node->prev;
		} else {
			node->next->prev = node->prev;
		}
	}

	QueueNode <TYPE> *first, *last, *curr, *del;
	unsigned int count;
};

#endif // _QUEUE_H_
