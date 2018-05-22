#pragma once

class ICoreBase
{
public:
	inline ICoreBase() { refCounter = 1; };
	
	// Adds a reference to the class
	virtual void AddRef() { refCounter++; };

	// Releases the object
	void Release();


protected:
	unsigned int refCounter;

	// Is called when no ref is last, override this method for your cleanup stuff
	virtual void finalRelease() = 0;

	virtual ~ICoreBase() {};
};