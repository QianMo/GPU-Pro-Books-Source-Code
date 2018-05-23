#ifndef FONT_H
#define FONT_H

/* return codes */
enum { FT_FAILURE = 0, FT_SUCCESS };
/* font size */
enum {
        FT_FONT_SIZE_SMALL  = 1<<3,
        FT_FONT_SIZE_MEDIUM = 1<<4,
        FT_FONT_SIZE_LARGE  = 1<<5
};

/* init / shutdown */
int ft_init(int gl_texture_unit);
int ft_shutdown();

/* print a string */
int ft_print(int font_size, int x, int y, const char *format, ...);

#endif
