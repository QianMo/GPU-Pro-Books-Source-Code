
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

#ifndef _TOKENIZER_H_
#define _TOKENIZER_H_

#include "../Platform.h"
#include "Array.h"

typedef bool (*BOOLFUNC)(const char ch);

bool isWhiteSpace(const char ch);
bool isNumeric(const char ch);
bool isAlphabetical(const char ch);
bool isNewLine(const char ch);

struct TokBuffer {
	char *buffer;
	unsigned int bufferSize;
};

class Tokenizer {
public:
	Tokenizer(unsigned int nBuffers = 1);
	~Tokenizer();

	void setString(const char *string);
	bool setFile(const char *fileName);
	void reset();

	bool goToNext(BOOLFUNC isAlpha = isAlphabetical);
	bool goToNextLine();
	char *next(BOOLFUNC isAlpha = isAlphabetical);
	char *nextAfterToken(const char *token, BOOLFUNC isAlpha = isAlphabetical);
	char *nextLine();
private:
	char *str;
	unsigned int length;
	unsigned int start, end;
	unsigned int capacity;

	unsigned int currentBuffer;
	Array <TokBuffer> buffers;

	char *getBuffer(unsigned int size);
};

#endif // _TOKENIZER_H_
