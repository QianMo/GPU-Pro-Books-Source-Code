/*
    Copyright (C) 2007-2009 by Jan Eric Kyprianidis <www.kyprianidis.com>
    All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
*/
#include "quicktime.h"
#include <cstdlib>
#include <string>
#include <cassert>
#ifdef WIN32
#include <QTML.h>
#include <Movies.h>
#else
#include <Quicktime/QTML.h>
#include <Quicktime/Movies.h>
#endif


bool quicktime_init() {
    OSErr err = noErr;
    #ifdef WIN32
    err = InitializeQTML(0);
    #endif
    if (err == noErr)
        err = EnterMovies();
    return (err == noErr);
}

void quicktime_exit() {
    ExitMovies();
    #ifdef WIN32
    TerminateQTML();
    #endif
}


struct quicktime_player::impl {
    Movie movie;
    Track track;
    Media media;
    int width;
    int height;
    GWorldPtr gworld;
    unsigned char* buffer;
    int playing;

    impl() {
        movie = NULL;
        width = height = 0;
        gworld = NULL;
        buffer = NULL;
        playing = 0;
    }

    ~impl() {
        if (movie) DisposeMovie(movie);
        if (gworld) DisposeGWorld(gworld);
        if (buffer) free(buffer);
    }
};


quicktime_player::quicktime_player(impl *p) : m(p) {}


quicktime_player::~quicktime_player() {
    delete m;
    m = NULL;
}


quicktime_player* quicktime_player::open( const char *path ) {
    OSErr err;
    OSType dataRefType;
    Handle dataRef = NULL;
    short resID = 0;
    impl *m = new impl;

    std::string pathStd = path;
    #ifdef WIN32
    for (std::string::iterator p = pathStd.begin(); p != pathStd.end(); ++p) if (*p == '/') *p = '\\';
    #endif
    
    CFStringRef cfPath = CFStringCreateWithCString(NULL, pathStd.c_str(), kCFStringEncodingISOLatin1);
    err = QTNewDataReferenceFromFullPathCFString(cfPath, (QTPathStyle)kQTNativeDefaultPathStyle, 0, &dataRef, &dataRefType);
    CFRelease(cfPath);
	if (err != noErr) {
		delete m;
        return NULL;
	}

    err = NewMovieFromDataRef(&m->movie, newMovieActive, &resID, dataRef, dataRefType);
    DisposeHandle(dataRef);
	if (err != noErr) {
		delete m;
        return NULL;
	}	

    m->track = GetMovieIndTrackType(m->movie, 1, VisualMediaCharacteristic, movieTrackCharacteristic);
    m->media = GetTrackMedia(m->track);

    Rect bounds;
    GetMovieBox(m->movie, &bounds);
    m->width = bounds.right;
    m->height = bounds.bottom;

    m->buffer = (unsigned char*)malloc(4 * m->width * m->height);
    err = QTNewGWorldFromPtr(&m->gworld, k32BGRAPixelFormat, &bounds, NULL, NULL, 0, m->buffer, 4 * m->width);
    if (err != noErr) {
        delete m;
        return NULL;
    }
    SetMovieGWorld(m->movie, m->gworld, NULL);

    return new quicktime_player(m);
}


int quicktime_player::get_width() {
    return m->width;
}


int quicktime_player::get_height() {
    return m->height;
}


void* quicktime_player::get_buffer() {
    return m->buffer;
}


bool quicktime_player::is_done() {
    return IsMovieDone(m->movie) != 0;
}


int quicktime_player::get_next_time(int current_time) {
    TimeValue next_time;
    OSType media = VideoMediaType;
    GetMovieNextInterestingTime(m->movie, nextTimeMediaSample, 1, &media, current_time, 0, &next_time, NULL);
    return next_time;
}


bool quicktime_player::is_playing() {
    return (m->playing > 0);
}


void quicktime_player::start() {
    if (m->playing == 0)
        StartMovie(m->movie);
    ++m->playing;
}


void quicktime_player::stop() {
    --m->playing;
    if (m->playing == 0)
        StopMovie(m->movie);
}


void quicktime_player::update() {
    if ((m->playing > 0) && (IsMovieDone(m->movie) != 0))
        GoToBeginningOfMovie(m->movie);
    MoviesTask(m->movie, 0);
    UpdateMovie(m->movie);
}


void quicktime_player::set_time(int current_time) {
    SetMovieTimeValue(m->movie, current_time);
    update();
}


void quicktime_player::goto_start() {
    GoToBeginningOfMovie(m->movie);
    update();
}


void quicktime_player::goto_end() {
    GoToEndOfMovie(m->movie);
    update();
}


int quicktime_player::get_time_scale() {
    return GetMovieTimeScale(m->movie);
}


int quicktime_player::get_time() {
    return GetMovieTime(m->movie, NULL);
}


int quicktime_player::get_duration() {
    return GetMovieDuration(m->movie);
}


int quicktime_player::get_frame_count() {
    return GetMediaSampleCount(m->media);
}


float quicktime_player::get_fps() {
	long sampleCount = GetMediaSampleCount(m->media);
	TimeScale timeScale = GetMediaTimeScale(m->media);
	TimeValue duration = GetMediaDuration(m->media);
    return float(sampleCount) * timeScale / duration;
}


struct quicktime_recorder::impl {
    int width;
    int height;
    DataHandler data_handler;
    Movie movie;
    Track track;
    Media media;
    GWorldPtr gworld;
    PixMapHandle pixmap;
    Handle handle;
    Ptr ptr;
    ImageDescriptionHandle image_desc;
    unsigned char* buffer;

    impl( int width_, int height_ ) 
        : width(width_), height(height_)
    {
        movie = NULL;
        data_handler = NULL;
        track = NULL;
        media = NULL;
        gworld = NULL;
        pixmap = NULL;
        handle = NULL;
        ptr = NULL;
        image_desc = NULL;
        buffer = NULL;
    }

    ~impl() {
        if (image_desc) DisposeHandle((Handle)image_desc);
        if (handle) DisposeHandle(handle);
        if (gworld) DisposeGWorld(gworld);
        if (buffer) delete[] buffer;
        if (movie) DisposeMovie( movie );
    }
};


quicktime_recorder::quicktime_recorder(impl *p) : m(p) {}


quicktime_recorder::~quicktime_recorder() {
    delete m;
    m = NULL;
}


quicktime_recorder* quicktime_recorder::create( const char *path2, int width, 
                                                int height, float fps ) {
    OSErr err;
    OSType   dataRefType;
    Handle   dataRef = NULL;
    impl *m = new impl(width, height);

    std::string pathStd = path2;
    #ifdef WIN32
    for (std::string::iterator p = pathStd.begin(); p != pathStd.end(); ++p) if (*p == '/') *p = '\\';    
    #endif

    CFStringRef cfPath = CFStringCreateWithCString(NULL, pathStd.c_str(), kCFStringEncodingISOLatin1);
    err = QTNewDataReferenceFromFullPathCFString(cfPath, kQTNativeDefaultPathStyle, 0, &dataRef, &dataRefType);
    CFRelease(cfPath);
    if (err != noErr) {
        delete m; 
        return NULL;
    }

    err = CreateMovieStorage(dataRef, dataRefType, FOUR_CHAR_CODE('TVOD'), smSystemScript, 
        createMovieFileDeleteCurFile | createMovieFileDontCreateResFile, &m->data_handler, &m->movie);
    DisposeHandle(dataRef);
    if (err != noErr) {
        delete m; 
        return NULL;
    }

    m->track = NewMovieTrack(m->movie, FixRatio(m->width, 1), FixRatio(m->height, 1), kNoVolume);
    err &= GetMoviesError();
	TimeScale timeScale = (TimeScale)(fps * 100.0f);
    m->media = NewTrackMedia(m->track, VideoMediaType, timeScale, nil, 0 );
    err &= GetMoviesError();
    if (err != noErr) {
        delete m; 
        return NULL;
    }
	SetMovieTimeScale(m->movie, timeScale);

    m->buffer = new unsigned char[4 * m->width * m->height];

    Rect rect;
    rect.left = rect.top = 0;
    rect.right = m->width;
    rect.bottom = m->height;

    err = QTNewGWorldFromPtr(&m->gworld, k32BGRAPixelFormat, &rect, NULL, NULL, 0, m->buffer, 4 * m->width);
    if (err != noErr) {
        delete m; 
        return NULL;
    }

    m->pixmap = GetGWorldPixMap(m->gworld);
    if (!m->pixmap) {
        delete m; 
        return NULL;
    }
    LockPixels(m->pixmap);

    long maxSize = 0;
    err = GetMaxCompressionSize(m->pixmap, &rect, 0, codecNormalQuality, kPNGCodecType, anyCodec, &maxSize); 
    if (err != noErr) {
        delete m; 
        return NULL;
    }

    m->handle = NewHandle(maxSize);
    if (!m->handle) {
        delete m; 
        return NULL;
    }

    HLockHi(m->handle);
    m->ptr = *m->handle;

    m->image_desc = (ImageDescriptionHandle)NewHandle(4);
    if (!m->image_desc) {
        delete m; 
        return NULL;
    }

    err = BeginMediaEdits( m->media );
    if (err != noErr) {
        delete m; 
        return NULL;
    }

    return new quicktime_recorder(m);
}


void quicktime_recorder::append_frame(int n) {
    Rect rect;
    rect.left = rect.top = 0;
    rect.right = m->width;
    rect.bottom = m->height;

    OSErr err = CompressImage(m->pixmap, &rect, codecNormalQuality, 
                              kPNGCodecType, m->image_desc, m->ptr);
    err = AddMediaSample(
        m->media, 
        m->handle, 
        0, 
        (**m->image_desc).dataSize, 
        100 * n, 
        (SampleDescriptionHandle)m->image_desc, 
        1, 
        0, 
        NULL
    );
}


void quicktime_recorder::finish() {
    EndMediaEdits( m->media );
	TimeValue duration = GetMediaDuration( m->media );
    InsertMediaIntoTrack( m->track, 0, 0, duration, fixed1 );

    OSErr err = UpdateMovieInStorage( m->movie, m->data_handler );
    err = CloseMovieStorage( m->data_handler );

    delete m;
}


void* quicktime_recorder::get_buffer() {
    return m->buffer;
}
