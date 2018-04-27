/******************************************************************************

 @File         dynamicArray.h

 @Title        Console Log

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Expanding array template class. Allows appending and direct
               access. Mixing access methods should be approached with caution.

******************************************************************************/
#ifndef DYNAMIC_ARRAY_H
#define DYNAMIC_ARRAY_H

#include "assert.h"
#include "../PVRTGlobal.h"

/*!****************************************************************************
Class
******************************************************************************/

/*!***************************************************************************
* @Class dynamicArray
* @Brief Expanding array template class.
* @Description Expanding array template class.
*****************************************************************************/
template<typename T>
class dynamicArray
{
public:
	static const int c_i32DefaultSize = 8;
	/*!***************************************************************************
	@Function			dynamicArray
	@Description		Blank constructor. Makes a default sized array.
	*****************************************************************************/
	dynamicArray():
	m_i32Size(0),
		m_i32Capacity(c_i32DefaultSize)
	{
		m_pArray = new T[c_i32DefaultSize];
	}

	/*!***************************************************************************
	@Function			dynamicArray
	@Input				i32Size	intial size of array
	@Description		Constructor taking initial size of array in elements.
	*****************************************************************************/
	dynamicArray(const int i32Size):
	m_i32Size(0),
		m_i32Capacity(i32Size)
	{
		_ASSERT(i32Size!=0);
		m_pArray = new T[i32Size];
	}

	/*!***************************************************************************
	@Function			dynamicArray
	@Input				original	the other dynamic array
	@Description		Copy constructor.
	*****************************************************************************/
	dynamicArray(const dynamicArray& original):
	m_i32Size(original.m_i32Size),
		m_i32Capacity(original.m_i32Capacity)
	{
		m_pArray = new T[m_i32Capacity];
		for(int i=0;i<m_i32Size;i++)
		{
			m_pArray[i]=original.m_pArray[i];
		}
	}

	/*!***************************************************************************
	@Function			dynamicArray
	@Input				pArray	an ordinary array
	@Input				i32Size		number of elements passed
	@Description		constructor from ordinary array.
	*****************************************************************************/
	dynamicArray(const T* const pArray, const unsigned int i32Size):
	m_i32Size(i32Size),
		m_i32Capacity(i32Size)
	{
		m_pArray = new T[i32Size];
		// copy old values to new array
		for(unsigned int i=0;i<m_i32Size;i++)
		{
			m_pArray[i]=pArray[i];
		}
	}

	/*!***************************************************************************
	@Function			~dynamicArray
	@Description		Destructor.
	*****************************************************************************/
	~dynamicArray()
	{
		if(m_pArray)
			delete[](m_pArray);
	}


	/*!***************************************************************************
	@Function			append
	@Input				addT	The element to append
	@Description		appends an element to the end of the array, expanding it
	if necessary.
	*****************************************************************************/
	void append(T addT)
	{
		if(m_i32Size==m_i32Capacity)		// should never be >
		{	// expand array: expensive

			m_i32Capacity*=2;	// array doubles in size for each expansion
			T *pDeadArray,*pNewArray = new T[m_i32Capacity]; // make new array

			// copy old values to new array
			for(int i=0;i<m_i32Size;i++)
			{
				pNewArray[i]=m_pArray[i];
			}


			pDeadArray = m_pArray;
			m_pArray = pNewArray;
			delete[](pDeadArray);
		}
		// add new element to end of array

		m_pArray[m_i32Size] = addT;
		m_i32Size++;
	}

	/*!***************************************************************************
	@Function			prepend
	@Input				addT	The element to prepend
	@Description		prepend an element to the beginning of the array, expanding
	it if necessary.
	*****************************************************************************/
	void prepend(T addT)
	{
		if(m_i32Size==m_i32Capacity)
		{	// expand array

			m_i32Capacity*=2;	// array doubles in size for each expansion
			T *pDeadArray,*pNewArray = new T[m_i32Capacity]; // make new array

			// copy all leaving a space at the start
			for(int i=0;i<m_i32Size;i++)
			{
				pNewArray[i+1]=m_pArray[i];
			}

			pDeadArray = m_pArray;
			m_pArray = pNewArray;
			delete[](pDeadArray);
		}
		else
		{
			// copy all leaving a space at the start
			for(int i=0;i<m_i32Size;i++)
			{
				m_pArray[i+1]=m_pArray[i];
			}
		}
		// add new element to end of array
		m_pArray[0] = addT;
		m_i32Size++;
	}

	/*!***************************************************************************
	@Function			=
	@Input				other	The dynamic array needing copied
	@Description		assignment operator.
	*****************************************************************************/
	template<typename T2>
		dynamicArray& operator=(const dynamicArray<T2>& other)
	{
		if(m_pArray)
			delete[](m_pArray);

		m_pArray = new T2[other.m_i32Capacity];
		for(int i=0;i<other.m_i32Size;i++)
		{
			m_pArray[i]=other.m_pArray[i];
		}
		m_i32Capacity = other.m_i32Capacity;
		m_i32Size = other.m_i32Size;
		return *this;
	}

	/*!***************************************************************************
	@Function			[]
	@Input				i32Index	index of element in array
	@Return				the element indexed
	@Description		indexed access into array. Note that this has no error
	checking whatsoever
	*****************************************************************************/
	T& operator[](const int i32Index)
	{
		_ASSERT(i32Index<m_i32Capacity);
		return m_pArray[i32Index];
	}

	/*!***************************************************************************
	@Function			[]
	@Input				i32Index	index of element in array
	@Return			the element indexed
	@Description		indexed access into array. Note that this has no error
	checking whatsoever
	*****************************************************************************/
	const T& operator[](const int i32Index) const
	{
		_ASSERT(i32Index<m_i32Capacity);
		return m_pArray[i32Index];
	}

	/*!***************************************************************************
	@Function			getSize
	@Return			size of array
	@Description		gives current size of array/number of elements
	*****************************************************************************/
	unsigned int getSize() const
	{
		return m_i32Size;
	}

	/*!***************************************************************************
	@Function			getCapacity
	@Return			capacity of array
	@Description		gives current allocated size of array/number of elements
	*****************************************************************************/
	unsigned int getCapacity() const
	{
		return m_i32Capacity;
	}

	/*!***************************************************************************
	@Function			expandToSize
	@Input				i32Size new capacity of array
	@Description		expands array to new capacity
	*****************************************************************************/
	void expandToSize(const int i32Size)
	{
		if(i32Size<=m_i32Capacity)
			return;	// nothing to be done

		T *pDeadArray,*pNewArray = new T[i32Size]; // make new array

		// copy old values to new array
		for(int i=0;i<m_i32Size;i++)
		{
			pNewArray[i]=m_pArray[i];
		}

		m_i32Capacity=i32Size;
		pDeadArray = m_pArray;
		m_pArray = pNewArray;
		delete[](pDeadArray);
	}

	/*!***************************************************************************
	@Function			bubbleSort
	@Description		very basic sorting function for array. Requires > operator
						to be defined.
						TODO: implement a better sorting algorithm
	*****************************************************************************/
	void bubbleSort()
	{
		bool bSwap;
		for(int i=0;i<m_i32Size;++i)
		{
			bSwap=false;
			for(unsigned int j=0;j<m_i32Size-1;++j)
			{
				if(m_pArray[j]>m_pArray[j+1])
				{
					PVRTswap(m_pArray[j],m_pArray[j+1]);
					bSwap = true;
				}
			}
			if(!bSwap)
				return;
		}
	}

	/*!***************************************************************************
	@Function			bubbleSort
	@Description		very basic sorting function for array. Requires > operator
	to be defined.
	use when T is a pointer so that the > operator is called
	on the actual type otherwise gets sorted by the pointer
	addresses - not so useful requires > operator to be defined
	TODO: implement a better sorting algorithm
	*****************************************************************************/
	void bubbleSortPointers()
	{
		bool bSwap;
		for(int i=0;i<m_i32Size;++i)
		{
			bSwap=false;
			for(int j=0;j<m_i32Size-1;++j)
			{
				if((*m_pArray[j])>(*m_pArray[j+1]))
				{
					PVRTswap(m_pArray[j],m_pArray[j+1]);
					bSwap = true;
				}
			}
			if(!bSwap)
				return;
		}
	}

private:
	int 	m_i32Size;				/*! current size of contents of array */
	int		m_i32Capacity;			/*! currently allocated size of array */
	T		*m_pArray;				/*! the actual array itself */
};







#endif // DYNAMIC_ARRAY_H

/*****************************************************************************
End of file (dynamicArray.h)
*****************************************************************************/
