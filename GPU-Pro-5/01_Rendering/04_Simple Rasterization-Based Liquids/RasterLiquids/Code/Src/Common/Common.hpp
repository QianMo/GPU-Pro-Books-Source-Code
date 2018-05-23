

#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <Common\Types.hpp>
#include <Common\Assert.hpp>
#include <cstring>

///<
namespace M {

	template<class UI>
	inline bool IsPowerOfTwo(UI _x){
		return ((_x != 0) && !(_x & (_x - 1)));
	}

    template<class T, class V>
    T RCast(V _pT){ return reinterpret_cast<T>(_pT); }

    template<class T, class V>
    T SCast(V _pT){ return static_cast<T>(_pT); }

	template<class V>
	int32 iCast(V _pT){ return static_cast<int32>(_pT); }

	template<class V>
	float32 fCast(V _pT){ return static_cast<float32>(_pT); }

    template<class T, class V>
    T DCast(V _pT){ return dynamic_cast<T>(_pT); }

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

///<
template<class T>
void DeleteArray(T** _ppToDelete)
{	
	if (_ppToDelete)
	{
		if(*_ppToDelete)
			delete[] *_ppToDelete;
		*_ppToDelete=0;
	}
}

///<
template<class T>
void Release(T* _pToRelease)
{
	if (_pToRelease)
		_pToRelease->Release();			
}

template<class T>
struct Releaser
{
	void operator()(T* _pT){M::Release(_pT);}
	void operator()(T& _T){_T.Release();}
};

///<
template<class T>
void Release(T** _ppToRelease)
{
	if (_ppToRelease)
	{
		Release(*_ppToRelease);
		*_ppToRelease=0;
	}
}


///<
template<class T>
void CopyConstChar(const char* _strSource, char** _strDest)
{
	if(*_strDest)
		M::DeleteArray(_strDest);

	int32 nameSize= strlen(_strSource)+1;
	*_strDest = new char[nameSize];
	memset(*_strDest,0,nameSize);
	memcpy(*_strDest,_strSource,nameSize);	

}





}

#endif

