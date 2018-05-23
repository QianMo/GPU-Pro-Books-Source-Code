/****************************************************************************

  GPU Pro 5 : Quaternions revisited - sample code
  All sample code written from scratch by Sergey Makeev specially for article.

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use the code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/

****************************************************************************/
#pragma once

#include <stdlib.h>

namespace Utils
{

	template<typename T>
	T Max(const T &a, const T &b)
	{
		return a < b ? b : a;
	}

	template<typename T>
	T Min(const T &a, const T &b)
	{
		return a < b ? a : b;
	}

	template<typename T>
	void Swap(T &a, T &b)
	{
		T tmp = a;
		a = b;
		b = tmp;
	}

	inline float FRand()
	{
		return ( rand() % RAND_MAX ) / ( float )( RAND_MAX );
	};
	
	inline float FRand( const float from, const float to )
	{
		return FRand() * ( to - from ) + from;
	};


	//Compile time is pow2 check
	//////////////////////////////////////////////////////////////////////////
	template< int N, int C = 1 >
	struct CompileTimeIsPow2Recurse
	{
		enum
		{
			result = CompileTimeIsPow2Recurse< N / 2, C * 2 >::result
		};
	};

	template< int C >
	struct CompileTimeIsPow2Recurse< 0, C >
	{
		enum
		{
			result = C
		};
	};


	template< int input_val >
	struct CompileTimeIsPow2
	{
		enum
		{
			result = CompileTimeIsPow2Recurse< input_val - 1 >::result == input_val ? 1 : 0
		};
	};
	//////////////////////////////////////////////////////////////////////////





	//////////////////////////////////////////////////////////////////////////
	template<typename T, int MAX_ELEMENTS_COUNT>
	class FixedArray
	{
		int elementsCount;
		
		unsigned char memoryBuffer[MAX_ELEMENTS_COUNT * sizeof(T)];
		T* data;

		void CopyFrom(const FixedArray & other)
		{
			clear();
			elementsCount = other.elementsCount;
			ASSERT(elementsCount >= 0 && elementsCount < MAX_ELEMENTS_COUNT, "Too much elements for FixedArray");

			for (int i = 0; i < elementsCount; i++)
			{
				data[i] = other.data[i];
			}
		}

	public:

		FixedArray()
		{
			data = (T*)&memoryBuffer[0];
			elementsCount = 0;
		}

		~FixedArray()
		{
			clear();
		}

		FixedArray( const FixedArray & other )
		{
			data = (T*)&memoryBuffer[0];
			elementsCount = 0;
			CopyFrom(other);
		}

		FixedArray & operator= ( const FixedArray & other )
		{
			CopyFrom(other);
			return *this;
		}

		void push_back(const T & element)
		{
			ASSERT(elementsCount < MAX_ELEMENTS_COUNT, "Too much elements for FixedArray");
			T* elementPtr = data + elementsCount;
			new(elementPtr) T(element);
			elementsCount++;
		}

		void clear()
		{
			trim(0);
		}

		void trim(int trimFromElementIndex)
		{
			for (int i = trimFromElementIndex; i < elementsCount; i++)
			{
				data[i].~T();
			}
			if (elementsCount > trimFromElementIndex)
			{
				elementsCount = trimFromElementIndex;
			}
		}

		int size() const
		{
			return elementsCount;
		}

		const T & operator [] (int index) const
		{
			ASSERT(index >= 0 && index < elementsCount, "FixedArray index is out of bounds");
			return data[index];
		}

		T & operator [] (int index)
		{
			ASSERT(index >= 0 && index < elementsCount, "FixedArray index is out of bounds");
			return data[index];
		}
	};


	//////////////////////////////////////////////////////////////////////////
	const char* StringFormat(const char* fmt, ...);
}
