

#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <Common/System/Assert.hpp>
///<
namespace M {
template<class T>
void Delete(T** _ppToDelete)
{
	if (_ppToDelete)
	{
		if(*_ppToDelete)
			delete *_ppToDelete;
		*_ppToDelete=0;
	}
}

template<class T>
void Release(T* _pToRelease)
{
	if (_pToRelease)
	{
		_pToRelease->Release();			
	}
}

///<
template<class T>
void Release(T** _ppToRelease)
{
	if (_ppToRelease)
	{
		if(*_ppToRelease)
		{
			(*_ppToRelease)->Release();
			*_ppToRelease=0;
		}	
	}
}

template<class T>
void Init(T* _pToRelease)
{
	if (_pToRelease)
		_pToRelease->Init();
}

}

#endif

