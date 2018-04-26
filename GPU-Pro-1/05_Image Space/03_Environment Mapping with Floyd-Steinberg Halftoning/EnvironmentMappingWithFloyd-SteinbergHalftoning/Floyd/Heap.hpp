// used for kd-tree building (kd-tree is for ray tracing acceleration)
// template class for a min-heap
// E must define operators for comparison

#pragma once 

template < class E >
class Heap
{
public:
	Heap ( int maxNumber);
    ~Heap ();

    inline bool insert ( const E &newElement ); 
    inline E removeMin ();
    inline E getMin ();

    inline int isEmpty () const;
    inline int isFull () const;

private:
    // Data members
	int maxSize,   // Maximum number of elements in the heap
        size;      // Actual number of elements in the heap
    E *element;   // Array containing the heap elements
};

template<class E>
Heap<E>::Heap( int maxNumber)
{
	maxSize = maxNumber;
    size = 0;
	element = new E [maxSize];
}

template <class E>
Heap<E>::~Heap()
{
	delete [] element;
}

template<class E>
bool Heap<E>::insert(const E & newElement)
{
	int currpos = size;
	int parentpos = (size -1)/2;
	int isPosition = 0;
	if ( isFull() )
    {
		return false;
    }
	// Inserts newElement into the heap;
	element[size] = newElement;
	size ++;
    
	// if newElement is lower, move it upward
	while ( currpos > 0 && !isPosition)
	{
		if ( element[currpos] >= element[parentpos] )
			isPosition = 1;
		else
		{
			element[currpos] = element[parentpos];
			element[parentpos] = newElement;
			currpos = parentpos;
			parentpos = ( currpos -1 )/2 ;
		}
	}
	return true;
}

template<class E>
E Heap<E>::getMin() 
{
	return element[0];
}

template<class E>
E Heap<E>::removeMin() 
{
	E delItem, temp;
	int currpos, lpos, rpos, isPosition = 0;
	if ( isEmpty() )
	{
		exit(1);
	}

	// removes the root
	delItem = element[0];
	size --;

	// replace the root with the bottom right-most element
	element[0] = element[size];
	temp = element[0];

	// set the current position and left and right child positions
	currpos = 0;
	lpos = 2*currpos + 1;
	rpos = 2*currpos + 2;

	// if the replacement is not proper, move it downward until the 
	// the properties that define a min-heap are restored
	while ( size > currpos+1 && !isPosition )
	{
        temp = element[currpos];         
		if ( rpos < size )
		{
			if ( element[currpos] > element[lpos] ||
				element[currpos] > element[rpos] )
			{ 
				if ( element[lpos] < element[rpos] )
				{
					element[currpos] = element[lpos];
					element[lpos] = temp;
					currpos = lpos;
				}
				else
				{
					element[currpos] = element[rpos];
					element[rpos] = temp;
					currpos = rpos;
				}
			}
		}
		else if ( lpos < size && element[lpos] < element[currpos] )
		{
			element[currpos] = element[lpos];
			element[lpos] = temp;
			currpos = lpos;
		}
		temp = element[currpos];
		lpos = 2*currpos +1;
		rpos = 2*currpos +2;

		if ( (lpos >= size)
			|| 
			element[currpos] <= element[lpos]
			&&
			element[currpos] <= element[rpos])
			isPosition = 1;
	}
	return delItem;
}

template<class E>
int Heap<E>:: isEmpty() const
{
	return size==0;
}

template<class E>
int Heap<E>::isFull() const
{
	return size == maxSize;
}

