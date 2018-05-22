
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

#include "Audio.h"
#include <stdio.h>

#include "codec.h"
#include "vorbisfile.h"

#ifdef _WIN32
#  pragma comment (lib, "../Framework3/Libs/OpenAL32.lib")
#  pragma comment (lib, "../Framework3/Libs/alut.lib")
#  pragma comment (lib, "../Framework3/Libs/ogg_static.lib")
#  pragma comment (lib, "../Framework3/Libs/vorbis_static.lib")
#  pragma comment (lib, "../Framework3/Libs/vorbisfile_static.lib")
#endif

Audio::Audio(){
	dev = alcOpenDevice(NULL);
	ctx = alcCreateContext(dev, NULL);
	alcMakeContextCurrent(ctx);

//	alDistanceModel(AL_INVERSE_DISTANCE);
	alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED);
	//alListenerf(AL_GAIN, 0.0f);
}

Audio::~Audio(){
	clear();

	alcMakeContextCurrent(NULL);
	alcDestroyContext(ctx);
	alcCloseDevice(dev);
}

void Audio::clear(){
	int index = sounds.getCount();
	while (index--){
		deleteSound(index);
	}

	index = soundSources.getCount();
	while (index--){
		deleteSoundSource(index);
	}
}

SoundID Audio::addSound(const char *fileName, unsigned int flags){
	Sound sound;

	// Clear error flag
	alGetError();

	const char *ext = strrchr(fileName, '.') + 1;
	char str[256];
	if (stricmp(ext, "ogg") == 0){
		FILE *file = fopen(fileName, "rb");
		if (file == NULL){
			sprintf(str, "Couldn't open \"%s\"", fileName);
			ErrorMsg(str);
			return SOUND_NONE;
		}

		OggVorbis_File vf;
		memset(&vf, 0, sizeof(vf));
		if (ov_open(file, &vf, NULL, 0) < 0){
			fclose(file);
			sprintf(str, "\"%s\" is not an ogg file", fileName);
			ErrorMsg(str);
			return SOUND_NONE;
		}

		vorbis_info *vi = ov_info(&vf, -1);

		int nSamples = (uint) ov_pcm_total(&vf, -1);
		int nChannels = vi->channels;
		sound.format = nChannels == 1? AL_FORMAT_MONO16 : AL_FORMAT_STEREO16;
		sound.sampleRate = vi->rate;

		sound.size = nSamples * nChannels;

		sound.samples = new short[sound.size];
		sound.size *= sizeof(short);

		int samplePos = 0;
		while (samplePos < sound.size){
			char *dest = ((char *) sound.samples) + samplePos;

			int bitStream, readBytes = ov_read(&vf, dest, sound.size - samplePos, 0, 2, 1, &bitStream);
			if (readBytes <= 0) break;
			samplePos += readBytes;
		}

		ov_clear(&vf);

	} else {
		ALboolean al_bool;
		ALvoid *data;
		alutLoadWAVFile(fileName, &sound.format, &data, &sound.size, &sound.sampleRate, &al_bool);
		sound.samples = (short *) data;
	}

	alGenBuffers(1, &sound.buffer);
	alBufferData(sound.buffer, sound.format, sound.samples, sound.size, sound.sampleRate);
	if (alGetError() != AL_NO_ERROR){
		alDeleteBuffers(1, &sound.buffer);

		sprintf(str, "Couldn't open \"%s\"", fileName);
		ErrorMsg(str);
		return SOUND_NONE;
	}

	return insertSound(sound);
}

SoundID Audio::insertSound(Sound &sound){
	for (uint i = 0; i < sounds.getCount(); i++){
		if (sounds[i].samples == NULL){
			sounds[i] = sound;
			return i;
		}
	}

	return sounds.add(sound);
}

void Audio::deleteSound(const SoundID sound){
	if (sounds[sound].samples){
		alDeleteBuffers(1, &sounds[sound].buffer);

		alutUnloadWAV(sounds[sound].format, sounds[sound].samples, sounds[sound].size, sounds[sound].sampleRate);
		//delete sound.samples;
		sounds[sound].samples = NULL;
	}
}

SoundSourceID Audio::addSoundSource(const SoundID sound, uint flags){
	SoundSrc soundSource;

	soundSource.sound = sound;

	alGenSources(1, &soundSource.source);
	alSourcei(soundSource.source, AL_LOOPING, (flags & LOOPING)? AL_TRUE : AL_FALSE);
	alSourcei(soundSource.source, AL_SOURCE_RELATIVE, (flags & RELATIVEPOS)? AL_TRUE : AL_FALSE);
	alSourcei(soundSource.source, AL_BUFFER, sounds[sound].buffer);
	alSourcef(soundSource.source, AL_MIN_GAIN, 0.0f);
	alSourcef(soundSource.source, AL_MAX_GAIN, 1.0f);

	return insertSoundSource(soundSource);
}

SoundID Audio::insertSoundSource(SoundSrc &source){
	for (uint i = 0; i < soundSources.getCount(); i++){
		if (soundSources[i].sound == SOUND_NONE){
			soundSources[i] = source;
			return i;
		}
	}

	return soundSources.add(source);
}

void Audio::deleteSoundSource(const SoundSourceID source){
	if (soundSources[source].sound != SOUND_NONE){
		alDeleteSources(1, &soundSources[source].source);
		soundSources[source].sound = SOUND_NONE;
	}
}

void Audio::play(const SoundSourceID source){
	alSourcePlay(soundSources[source].source);
}

void Audio::stop(const SoundSourceID source){
	alSourceStop(soundSources[source].source);
}

void Audio::pause(const SoundSourceID source){
	alSourcePause(soundSources[source].source);
}

bool Audio::isPlaying(const SoundSourceID source){
	ALint state;
	alGetSourcei(soundSources[source].source, AL_SOURCE_STATE, &state);

	return (state == AL_PLAYING);
}

void Audio::setListenerOrientation(const vec3 &position, const vec3 &zDir){
	alListenerfv(AL_POSITION, position);

	float orient[] = { zDir.x, zDir.y, zDir.z, 0, -1, 0 };
	alListenerfv(AL_ORIENTATION, orient);
}

void Audio::setSourceGain(const SoundSourceID source, const float gain){
	alSourcef(soundSources[source].source, AL_GAIN, gain);
}

void Audio::setSourcePosition(const SoundSourceID source, const vec3 &position){
	alSourcefv(soundSources[source].source, AL_POSITION, position);
}

void Audio::setSourceAttenuation(const SoundSourceID source, const float rollOff, const float refDistance){
	alSourcef(soundSources[source].source, AL_REFERENCE_DISTANCE, refDistance);
	alSourcef(soundSources[source].source, AL_ROLLOFF_FACTOR, rollOff);
}
