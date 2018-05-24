#pragma once


#define SAFE_CALL(x) if (x) x


#define SAFE_DELETE(x) \
	if (x) \
	{ \
		delete x; \
		x = nullptr; \
	}


#define SAFE_DELETE_ARRAY(x) \
	if (x) \
	{ \
		delete[] x; \
		x = nullptr; \
	}


#define SAFE_RELEASE(x) \
	if (x) \
	{ \
		x->Release(); \
		x = nullptr; \
	}


#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))


#define BUFFER_OFFSET(offset) ((char*)nullptr + offset)
