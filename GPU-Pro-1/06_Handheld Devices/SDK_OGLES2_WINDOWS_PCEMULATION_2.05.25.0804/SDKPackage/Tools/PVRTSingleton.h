/******************************************************************************

 @File         PVRTSingleton.h

 @Title        PVRTSingleton

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Singleton template Pattern Usage: Inherit from CPVRTSingleton
               class like this: class Foo : public CPVRTSingleton<Foo> { ... };

******************************************************************************/
#ifndef __PVRTSINGLETON__
#define __PVRTSINGLETON__

template<typename T> class CPVRTSingleton
{
private:
	CPVRTSingleton(const CPVRTSingleton&);
	CPVRTSingleton & operator=(const CPVRTSingleton&);

public:
	static T& inst()
	{
		static T object;
		return object;
	}

	static T* ptr()
	{
		return &inst();
	}

protected:
	CPVRTSingleton() {};
	virtual ~CPVRTSingleton() {};
};


#endif // __PVRTSINGLETON__

/*****************************************************************************
End of file (PVRTSingleton.h)
*****************************************************************************/
