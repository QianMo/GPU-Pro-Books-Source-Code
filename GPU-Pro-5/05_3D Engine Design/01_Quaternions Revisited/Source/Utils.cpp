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
#include <stdio.h>
#include <stdarg.h>
#include "Utils.h"


const char* Utils::StringFormat(const char* fmt, ...)
{
	const int FORMATTER_BUFFER_SIZE = 65536;
	static char formatterBuffer[FORMATTER_BUFFER_SIZE] = { '\0' };

	va_list va;
	va_start( va, fmt );
	_vsnprintf_s( formatterBuffer, FORMATTER_BUFFER_SIZE, FORMATTER_BUFFER_SIZE - 1, fmt, va );
	va_end( va );
	return formatterBuffer;
}