#define nr_chrs_S 104
#define firstchr_S 32
#define charheight_S 8

#ifdef __cplusplus
extern "C" {
#endif
    extern unsigned char lentbl_S[104];
    extern unsigned char * chrtbl_S[104];

    // Draws the character to the destination buffer with given pitch in the given rgba color
    // returns number of columns drawn
    int DrawChar(unsigned char *rgbaBuff, int pitch, int theChar, unsigned int color);

#ifdef __cplusplus
}
#endif
