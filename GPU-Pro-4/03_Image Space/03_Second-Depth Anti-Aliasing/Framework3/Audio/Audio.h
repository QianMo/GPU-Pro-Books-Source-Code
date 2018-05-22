
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

#ifndef _AUDIO_H_
#define _AUDIO_H_

#include "../Platform.h"
#include "../Math/Vector.h"
#include "../Util/Array.h"

#include "al.h"
#include "alc.h"
#include "alut.h"

typedef int SoundID;
typedef int SoundSourceID;

#define SOUND_NONE (-1)

// Sound source flags
#define LOOPING 0x1
#define RELATIVEPOS 0x2

struct Sound {
	ALenum format;
	ALuint buffer;

	short *samples;
	int sampleRate;
	int size;
};

struct SoundSrc {
	ALuint source;
	SoundID sound;
};


class Audio {
public:
	Audio();
	~Audio();

	void clear();

	SoundID addSound(const char *fileName, uint flags = 0);
	void deleteSound(const SoundID sound);

	SoundSourceID addSoundSource(const SoundID sound, uint flags = 0);
	void deleteSoundSource(const SoundSourceID source);

	void play(const SoundSourceID source);
	void stop(const SoundSourceID source);
	void pause(const SoundSourceID source);
	bool isPlaying(const SoundSourceID source);

	void setListenerOrientation(const vec3 &position, const vec3 &zDir);

	void setSourceGain(const SoundSourceID source, const float gain);
	void setSourcePosition(const SoundSourceID source, const vec3 &position);
	void setSourceAttenuation(const SoundSourceID source, const float rollOff, const float refDistance);

protected:
	SoundID insertSound(Sound &sound);
	SoundID insertSoundSource(SoundSrc &source);


	ALCcontext *ctx;
	ALCdevice *dev;

	Array <Sound> sounds;
	Array <SoundSrc> soundSources;
};

#endif // _AUDIO_H_
