#include "PlatformDll.h"

#ifdef EXP_IMP
#undef EXP_IMP
#endif

#ifdef MD_ENGINE_PROJECT
#define EXP_IMP DLLEXPORT
#else
#define EXP_IMP DLLIMPORT
#endif
