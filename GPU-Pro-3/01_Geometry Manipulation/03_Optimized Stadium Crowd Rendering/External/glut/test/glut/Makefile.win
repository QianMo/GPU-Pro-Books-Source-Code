# Makefile for Win32

!include <win32.mak>

TOP  = ../..
SRCS = bigtest.c cursor_test.c keyup_test.c over_test.c menu_test.c shape_test.c test1.c test10.c test11.c test12.c test13.c test14.c test15.c test16.c test17.c test18.c test19.c test2.c test20.c test21.c test22.c test23.c test24.c test3.c test4.c test5.c test6.c test7.c test8.c test9.c timer_test.c test25.c test26.c test27.c test28.c joy_test.c

!include "$(TOP)/glutwin32.mak"

# BUG ALERT: In Win32 GLUT 3.6, test18 and test22 do not pass.  To be fixed.

# NOTE: first run of test2 can fail since time dependent, but run twice

test : $(TARGETS)
	-.\test2
	.\test2
	.\test3
	.\test4
	.\test5
	.\test7
	.\test8
	.\test9
	.\test10
	.\test11
	.\test12
	.\test13
	.\test14
	.\test15
	.\test16
	.\test17
	.\test19
	.\test20
	.\test21
	.\test23
	.\test24
	.\test25
	.\test26
	.\test27
	.\test28
