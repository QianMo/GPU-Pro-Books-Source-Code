#ifndef _AL_ALUT_H
#define _AL_ALUT_H

#include "al.h"

#if defined(_WIN32) && !defined(_XBOX) && 0
 #if defined (_OPENAL32LIB)
  #define ALUTAPI __declspec(dllexport)
 #else
  #define ALUTAPI __declspec(dllimport)
 #endif
#else
 #define ALUTAPI extern
#endif

#if defined(_WIN32)
 #define ALUTAPIENTRY __cdecl
#else
 #define ALUTAPIENTRY
#endif

#if TARGET_OS_MAC
 #pragma export on
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(ALUT_NO_PROTOTYPES)

ALUTAPI void ALUTAPIENTRY alutInit( int *argc, char *argv[] );
ALUTAPI void ALUTAPIENTRY alutExit( void );

#if defined(MACINTOSH_AL)
/* Windows and Linux versions have a loop parameter, Macintosh doesn't */
ALUTAPI void ALUTAPIENTRY alutLoadWAVFile( const ALbyte *file, ALenum *format, ALvoid **data, ALsizei *size, ALsizei *freq );
ALUTAPI void ALUTAPIENTRY alutLoadWAVMemory( const ALbyte *memory, ALenum *format, ALvoid **data, ALsizei *size, ALsizei *freq );
#else
ALUTAPI void ALUTAPIENTRY alutLoadWAVFile( const ALbyte *file, ALenum *format, ALvoid **data, ALsizei *size, ALsizei *freq, ALboolean *loop );
ALUTAPI void ALUTAPIENTRY alutLoadWAVMemory( const ALbyte *memory, ALenum *format, ALvoid **data, ALsizei *size, ALsizei *freq, ALboolean *loop );
#endif
ALUTAPI void ALUTAPIENTRY alutUnloadWAV( ALenum format, ALvoid *data, ALsizei size, ALsizei freq );

#else /* ALUT_NO_PROTOTYPES */

ALUTAPI void (ALUTAPIENTRY *alutInit)( int *argc, char *argv[] );
ALUTAPI void (ALUTAPIENTRY *alutExit)( void );
#if defined(MACINTOSH_AL)
ALUTAPI void (ALUTAPIENTRY *alutLoadWAVFile)( const ALbyte *file, ALenum *format, ALvoid **data, ALsizei *size, ALsizei *freq );
ALUTAPI void (ALUTAPIENTRY *alutLoadWAVMemory)( const ALbyte *memory, ALenum *format, ALvoid **data, ALsizei *size, ALsizei *freq );
#else
ALUTAPI void (ALUTAPIENTRY *alutLoadWAVFile)( const ALbyte *file, ALenum *format, ALvoid **data, ALsizei *size, ALsizei *freq, ALboolean *loop );
ALUTAPI void (ALUTAPIENTRY *alutLoadWAVMemory)( const ALbyte *memory, ALenum *format, ALvoid **data, ALsizei *size, ALsizei *freq, ALboolean *loop );
#endif
ALUTAPI void (ALUTAPIENTRY *alutUnloadWAV)( ALenum format, ALvoid *data, ALsizei size, ALsizei freq );

#endif /* ALUT_NO_PROTOTYPES */

#if TARGET_OS_MAC
 #pragma export off
#endif

#if defined(__cplusplus)
}
#endif

#endif
