/*
    Copyright (C) 2007-2009 Jan Eric Kyprianidis <www.kyprianidis.com>
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
#ifndef INCLUDED_QUICKTIME_H
#define INCLUDED_QUICKTIME_H

extern bool quicktime_init();
extern void quicktime_exit();

class quicktime_player {
public:
    static quicktime_player* open( const char *path );
    ~quicktime_player();

    void    start           ();
    void    stop            ();
    void    update          ();
    void    set_time        (int current_time);
    void    goto_start      ();
    void    goto_end        ();
    
    bool    is_valid        ();
    int     get_width       ();
    int     get_height      ();
    void*   get_buffer      ();
    bool    is_done         ();
    int     get_next_time   (int current_time);
    bool    is_playing      ();
    int     get_time_scale  ();
    int     get_time        ();
    int     get_duration    ();
    int     get_frame_count ();
    float   get_fps         ();
    
private:
    struct impl;
    impl *m;
    quicktime_player(impl*);
};

class quicktime_recorder {
public:
    static quicktime_recorder* create( const char *path, int width, int height, float fps );
    ~quicktime_recorder();

    void    append_frame  (int n = 1);
    void    finish        ();
    void*   get_buffer    ();

private:
    struct impl;
    impl *m;
    quicktime_recorder(impl*);
};

#endif
