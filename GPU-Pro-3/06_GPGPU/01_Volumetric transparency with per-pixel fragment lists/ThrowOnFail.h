#pragma once

class HrException
{
public:
	char* errorMessage;
	char* filename;
	int lineNumber;
	HRESULT hr;

	HrException(HRESULT hr, char* errorMessage, char* filename, int lineNumber)
		:hr(hr), errorMessage(errorMessage), filename(filename), lineNumber(lineNumber)
	{
	}
};

class ThrowOnFail
{
	char* errorMessage;
	char* filename;
	int lineNumber;
public:
	ThrowOnFail(char* errorMessage, char* filename, int lineNumber)
		:errorMessage(errorMessage), filename(filename), lineNumber(lineNumber)
    {
    }

    void operator=(HRESULT hr)
    {
        if(FAILED(hr))
            throw HrException(hr, errorMessage, filename, lineNumber);
    }
};

