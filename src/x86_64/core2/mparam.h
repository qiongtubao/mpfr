/* Various Thresholds of MPFR, not exported.  -*- mode: C -*-

Copyright 2005-2024 Free Software Foundation, Inc.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
https://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

/* Generated by MPFR's tuneup.c, 2018-02-22, gcc 6.3.0 */
/* gcc14.fsffrance.org (Intel(R) Xeon(R) CPU X5450 @ 3.00GHz) with gmp 6.1.2 */

#define MPFR_MULHIGH_TAB  \
 -1,0,-1,0,0,0,0,0,0,0,0,0,9,9,9,9, \
 10,10,11,12,13,12,13,14,15,16,17,18,17,18,19,20, \
 23,24,24,24,24,26,26,28,28,24,24,24,28,30,28,28, \
 32,32,30,32,32,34,36,36,36,34,38,38,40,38,40,40, \
 48,48,46,48,48,48,48,48,48,48,48,48,48,52,56,56, \
 56,56,56,56,56,60,60,60,64,56,56,64,64,60,60,60, \
 64,64,64,64,75,64,75,64,64,69,75,75,64,81,84,84, \
 80,81,81,80,81,81,81,84,87,87,87,87,84,92,87,81, \
 81,90,93,92,93,93,87,90,90,93,92,93,93,93,92,93, \
 92,93,104,93,105,93,99,105,105,104,105,108,105,105,108,105, \
 105,108,110,111,111,110,111,114,117,114,117,116,105,117,116,117, \
 141,141,141,141,141,141,141,141,140,141,141,141,141,141,141,141, \
 140,141,141,141,141,141,141,141,141,140,141,141,141,153,140,140, \
 141,141,141,141,141,141,165,165,165,165,165,153,165,165,165,165, \
 153,165,165,165,165,177,165,188,165,165,188,165,165,188,165,165, \
 188,188,165,188,188,188,188,188,188,188,188,188,188,188,186,188, \
 188,188,188,188,188,188,188,188,188,188,188,188,188,188,188,204, \
 204,188,204,204,204,204,204,202,204,204,204,204,220,220,220,208, \
 203,204,204,220,220,220,220,220,220,220,220,220,220,220,220,220, \
 220,220,220,236,236,236,236,236,236,236,236,236,236,236,236,236, \
 236,236,236,236,236,236,236,236,236,236,236,282,282,282,282,282, \
 282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282, \
 282,282,282,282,282,282,282,282,282,282,282,282,282,282,282,282, \
 282,282,282,282,282,282,282,282,281,282,306,282,282,282,306,282, \
 282,282,330,329,330,330,330,330,306,330,306,330,306,306,330,306, \
 306,330,330,330,330,330,330,330,330,330,330,330,330,330,330,330, \
 330,330,330,330,329,330,330,330,330,330,360,360,330,330,360,360, \
 360,330,360,360,360,360,360,360,360,360,360,360,368,368,360,376, \
 376,376,368,376,368,376,376,376,376,376,376,376,376,376,368,368, \
 376,376,376,376,376,376,375,376,376,376,376,376,368,368,376,376, \
 408,408,408,408,408,376,408,408,376,375,376,439,376,440,440,376, \
 408,376,376,376,376,376,408,376,440,440,440,440,440,440,440,440, \
 440,440,432,432,440,440,440,440,440,440,439,440,440,440,440,439, \
 439,440,408,440,440,440,439,440,440,440,440,440,440,440,472,440, \
 440,472,472,440,440,440,440,440,440,440,440,439,440,440,440,440, \
 440,440,440,440,440,439,440,440,440,440,440,440,440,440,440,440, \
 440,440,440,440,440,472,439,440,440,440,472,472,472,472,472,472, \
 472,470,472,472,504,471,472,472,480,472,472,472,472,472,472,503, \
 472,472,472,472,472,472,472,472,504,503,504,504,504,504,504,504, \
 472,504,504,504,504,504,504,504,504,504,504,504,504,496,504,504, \
 504,504,504,504,504,504,504,504,504,504,504,504,504,504,504,504, \
 504,504,504,504,504,504,504,504,544,544,544,544,544,544,544,544, \
 544,536,544,544,544,544,544,544,544,544,544,544,544,544,544,544, \
 544,544,544,544,544,544,544,544,544,544,592,592,592,592,592,592, \
 591,592,592,591,592,592,568,568,592,591,592,592,592,592,592,640, \
 592,592,592,592,592,592,592,592,592,592,592,591,592,592,592,592, \
 592,592,592,624,624,623,630,631,632,632,632,632,632,592,632,639, \
 640,640,640,639,640,640,640,640,640,632,640,632,632,632,632,632, \
 640,632,632,639,632,632,632,640,639,640,640,640,664,664,640,640, \
 640,640,640,664,664,664,664,664,664,664,640,663,640,640,664,664, \
 640,664,640,664,664,664,664,664,640,664,688,688,688,687,688,688, \
 680,640,664,664,664,664,688,664,664,688,664,688,640,664,664,664, \
 639,688,640,640,640,640,712,640,711,712,712,736,688,712,712,712, \
 736,664,664,664,688,664,664,664,728,664,728,728,728,728,736,736, \
 736,735,736,736,736,736,728,728,735,736,736,736,736,736,736,736, \
 735,736,735,736,728,736,736,728,728,735,736,735,736,736,736,736, \
 736,736,735,736,735,736,736,735,736,728,728,728,728,736,736,735, \
 735,736,736,736,736,736,736,728,735,736,736,736,736,736,736,736, \
 735,736,736,736,736,736,736,736,736,736,736,736,736,735,734,736, \
 736,735,736,736,736,736,736,736,735,736,736,735,736,736,735,735, \
 736,736,736,784,736,735,784,784,784,784,736,736,824,783,736,735, \
 783,784,784,736,784,824,784,784,784,784,784,784,832,736,735,832, \
 824,784,736,736,824,736,824,784,784,784,784,832,832,784,824,824, \
 824,856,822,824,824,824,824,824,824,824,824,824,824,824,824,824 \

#define MPFR_SQRHIGH_TAB  \
 -1,-1,-1,-1,-1,-1,-1,-1,6,6,7,7,8,9,9,9, \
 10,10,11,12,13,14,13,14,15,16,17,17,17,18,20,20, \
 21,22,19,19,20,21,24,25,22,22,23,24,24,25,26,27, \
 26,27,27,27,34,34,30,34,34,34,34,34,38,38,38,38, \
 38,42,40,40,44,46,44,38,40,46,42,42,44,44,46,46, \
 46,48,48,48,48,48,50,48,48,50,48,48,50,60,60,60, \
 50,60,52,60,60,60,60,60,60,64,68,64,64,68,64,64, \
 68,64,64,68,68,68,68,72,68,68,76,76,76,76,76,76, \
 76,80,80,80,80,80,80,80,80,76,80,84,84,88,84,84, \
 76,80,76,80,80,84,80,84,84,84,88,88,88,92,91,92, \
 92,92,96,96,96,88,92,92,92,92,96,96,96,92,96,96, \
 95,105,96,110,111,111,117,111,117,117,117,117,117,117,117,123, \
 123,117,123,117,117,117,117,117,116,129,135,129,123,135,135,123, \
 123,129,129,135,135,135,134,135,129,135,140,135,140,141,141,141, \
 141,141,141,141,141,141,141,141,140,141,140,140,141,141,141,141, \
 141,140,147,140,141,141,141,153,147,153,147,147,172,141,140,141, \
 140,164,172,164,171,164,172,172,141,141,172,180,180,188,180,180, \
 180,180,188,180,188,188,188,188,187,188,188,180,188,172,188,188, \
 187,188,187,188,188,180,172,180,180,196,179,180,188,188,188,187, \
 188,188,186,185,188,188,188,188,188,196,188,188,188,196,188,188, \
 188,188,188,196,195,188,188,188,196,196,187,188,188,188,188,196, \
 195,195,195,196,195,196,204,204,212,212,204,188,196,196,196,204, \
 212,212,212,212,212,204,234,234,234,258,234,246,234,246,234,234, \
 246,246,246,234,246,258,258,246,246,234,258,258,258,258,258,246, \
 258,270,258,258,258,258,258,270,258,270,257,258,270,258,258,270, \
 270,282,270,282,282,270,282,281,282,282,282,282,282,282,282,282, \
 282,282,270,282,282,282,282,282,282,282,282,282,282,282,282,282, \
 282,282,282,282,282,282,282,282,282,282,282,282,282,282,312,312, \
 282,312,312,312,312,312,312,312,312,328,312,282,282,328,282,270, \
 282,282,328,282,328,328,328,328,282,328,344,282,282,282,328,344, \
 328,344,344,344,344,360,344,344,360,344,360,344,344,344,344,344, \
 344,360,360,344,344,360,360,344,344,360,344,344,360,360,360,360, \
 360,360,328,360,328,360,328,344,360,328,328,360,344,360,360,344, \
 360,360,344,360,344,344,360,344,360,344,344,344,344,360,344,344, \
 360,360,360,359,360,360,360,360,360,360,360,360,360,360,359,360, \
 360,360,360,360,360,360,359,359,360,360,360,344,360,360,360,360, \
 360,360,360,360,360,360,360,360,391,360,392,360,360,359,360,359, \
 360,360,360,360,360,360,360,360,360,360,359,360,360,360,424,424, \
 423,424,424,424,472,424,472,424,424,424,424,424,424,471,472,472, \
 424,472,472,424,472,472,472,472,472,424,424,472,472,424,424,472, \
 472,424,472,472,472,472,472,472,472,472,472,471,472,472,472,472, \
 472,472,472,472,472,472,471,472,471,472,472,472,472,472,472,472, \
 472,471,472,472,472,504,504,472,472,472,472,472,504,504,472,504, \
 472,504,504,504,472,504,504,472,472,472,471,472,472,472,472,471, \
 472,472,472,472,472,472,472,472,472,472,471,472,472,504,504,472, \
 472,504,472,536,472,472,471,472,472,472,536,472,472,536,536,536, \
 536,536,536,536,504,504,504,536,536,504,536,536,504,536,504,504, \
 536,504,568,504,568,568,568,568,568,568,568,568,568,568,568,568, \
 568,567,568,536,568,536,568,568,568,536,568,568,536,536,536,536, \
 536,568,536,536,536,536,568,536,536,568,536,536,568,536,536,568, \
 568,568,568,568,568,568,568,568,568,568,567,568,568,568,568,568, \
 568,568,568,568,568,568,567,568,567,568,568,600,600,568,568,568, \
 568,600,600,568,600,600,568,600,600,600,568,600,600,600,600,600, \
 600,600,600,600,600,632,599,600,600,600,600,600,600,600,600,600, \
 632,600,568,600,568,600,568,568,632,568,631,600,568,568,568,568, \
 632,632,632,632,624,632,631,600,624,632,632,600,600,632,632,632, \
 600,600,600,600,600,600,632,600,632,600,600,600,664,664,632,664, \
 664,600,568,600,664,568,600,664,568,568,600,600,568,632,632,632, \
 599,600,568,600,632,632,632,632,600,600,568,600,600,632,632,600, \
 600,632,599,600,600,632,599,600,600,600,600,600,600,600,664,600, \
 600,664,600,664,632,696,696,696,696,696,696,696,695,696,696,695, \
 696,696,632,632,632,632,632,632,632,632,632,632,632,632,632,631, \
 632,632,632,632,632,631,728,728,632,632,696,632,696,696,696,664, \
 696,696,696,664,662,664,664,664,664,664,664,664,664,664,664,664 \

#define MPFR_DIVHIGH_TAB  \
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, /*0-15*/ \
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, /*16-31*/ \
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, /*32-47*/ \
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, /*48-63*/ \
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, /*64-79*/ \
 0,0,0,0,0,0,0,0,48,0,0,0,0,0,0,50, /*80-95*/ \
 54,55,0,52,56,52,56,56,58,55,58,60,59,60,64,59, /*96-111*/ \
 64,59,62,68,60,60,64,64,68,62,64,66,72,68,66,68, /*112-127*/ \
 68,68,68,68,71,69,74,76,70,78,72,72,72,72,78,74, /*128-143*/ \
 77,75,78,78,78,78,84,77,84,79,80,79,80,86,92,92, /*144-159*/ \
 83,83,92,92,96,96,96,92,96,96,96,92,92,104,96,104, /*160-175*/ \
 92,104,96,104,96,96,96,104,104,104,112,104,104,104,112,116, /*176-191*/ \
 112,112,112,104,112,120,116,112,116,120,112,112,118,104,119,120, /*192-207*/ \
 112,120,124,124,112,128,112,120,116,116,128,112,120,128,116,120, /*208-223*/ \
 128,116,120,120,120,128,120,128,128,120,128,128,128,128,124,128, /*224-239*/ \
 128,124,128,128,128,128,128,128,128,128,128,128,135,136,136,132, /*240-255*/ \
 136,136,132,134,135,136,137,149,135,160,136,136,156,150,160,160, /*256-271*/ \
 144,160,144,148,160,160,161,144,144,156,160,156,156,160,160,160, /*272-287*/ \
 160,150,162,148,160,150,158,158,160,174,156,160,164,160,160,162, /*288-303*/ \
 156,160,160,168,160,168,158,172,160,159,160,184,162,192,184,174, /*304-319*/ \
 184,173,168,192,185,186,184,184,184,184,192,192,168,184,184,191, /*320-335*/ \
 172,184,184,185,192,186,185,184,185,186,192,184,184,184,184,192, /*336-351*/ \
 184,182,192,191,185,189,192,184,192,192,192,184,184,208,192,185, /*352-367*/ \
 192,216,208,216,208,192,192,208,192,192,192,192,192,208,208,224, /*368-383*/ \
 216,216,208,207,222,208,224,224,208,216,232,224,208,208,208,208, /*384-399*/ \
 232,208,224,224,224,208,216,224,228,216,208,224,232,240,224,222, /*400-415*/ \
 224,224,224,224,224,216,232,224,232,232,216,216,232,224,232,232, /*416-431*/ \
 240,240,224,224,224,224,232,228,246,240,232,232,240,240,239,240, /*432-447*/ \
 240,240,232,232,240,240,240,256,256,240,240,256,240,256,253,240, /*448-463*/ \
 256,256,256,240,255,256,240,256,256,264,256,240,240,240,256,256, /*464-479*/ \
 256,256,248,256,257,256,256,256,256,256,256,256,264,256,256,256, /*480-495*/ \
 256,256,256,256,256,288,256,270,288,256,288,256,270,256,276,288, /*496-511*/ \
 272,270,282,288,276,276,270,288,288,288,288,276,276,288,288,288, /*512-527*/ \
 288,288,280,288,276,272,288,288,288,276,272,280,288,312,276,288, /*528-543*/ \
 284,288,288,312,288,288,288,288,312,316,288,306,288,288,312,312, /*544-559*/ \
 288,312,288,312,320,304,305,288,312,320,320,312,336,288,312,312, /*560-575*/ \
 312,330,312,312,330,320,300,336,312,320,312,312,312,312,320,336, /*576-591*/ \
 316,312,312,312,320,312,348,322,318,312,336,336,312,320,336,312, /*592-607*/ \
 316,324,336,312,336,384,312,324,384,368,336,320,384,320,336,384, /*608-623*/ \
 384,320,316,336,320,368,384,384,368,330,384,368,384,336,384,368, /*624-639*/ \
 384,368,384,336,384,336,336,336,384,336,336,384,384,384,384,370, /*640-655*/ \
 384,368,384,368,384,384,384,384,368,354,359,368,336,368,368,368, /*656-671*/ \
 384,384,384,368,384,384,368,384,384,376,368,384,369,384,384,368, /*672-687*/ \
 372,384,384,384,384,384,368,368,384,368,368,384,383,368,369,368, /*688-703*/ \
 370,368,368,369,416,384,384,384,384,384,384,384,368,384,384,384, /*704-719*/ \
 384,416,416,384,384,369,368,416,368,368,384,384,384,384,384,384, /*720-735*/ \
 384,384,384,384,384,384,384,384,384,384,384,384,382,376,384,408, /*736-751*/ \
 384,384,384,382,408,384,384,416,384,384,416,384,384,384,448,416, /*752-767*/ \
 416,432,416,416,416,416,416,416,416,416,416,432,432,416,408,432, /*768-783*/ \
 432,416,416,416,416,416,416,416,432,416,432,416,432,416,416,416, /*784-799*/ \
 432,440,416,432,432,416,448,416,416,416,448,416,416,416,448,464, /*800-815*/ \
 464,432,432,416,448,416,448,464,448,464,464,464,440,456,440,448, /*816-831*/ \
 480,472,480,432,432,464,448,432,480,464,480,432,464,480,432,480, /*832-847*/ \
 432,440,432,432,448,440,480,464,480,480,440,448,480,448,448,440, /*848-863*/ \
 448,464,448,466,448,448,480,448,448,448,512,466,480,480,480,456, /*864-879*/ \
 464,448,464,464,464,448,464,464,512,464,512,480,480,480,464,480, /*880-895*/ \
 480,480,480,480,512,480,480,480,480,456,464,464,466,464,512,464, /*896-911*/ \
 480,480,512,464,512,468,512,472,512,466,480,464,512,480,512,480, /*912-927*/ \
 480,480,510,512,480,480,480,480,480,512,480,562,512,480,512,480, /*928-943*/ \
 512,512,480,480,480,538,480,480,480,528,540,480,564,512,540,512, /*944-959*/ \
 540,512,564,512,512,512,564,550,564,512,512,512,561,512,563,563, /*960-975*/ \
 512,512,512,512,564,512,512,512,564,564,512,564,563,564,512,540, /*976-991*/ \
 562,540,528,512,540,512,552,512,552,560,512,512,552,564,540,552, /*992-1007*/ \
 540,561,562,552,576,564,540,562,552,562,563,562,564,562,564,564 /*1008-1023*/ \

#define MPFR_MUL_THRESHOLD 10 /* limbs */
#define MPFR_SQR_THRESHOLD 13 /* limbs */
#define MPFR_DIV_THRESHOLD 5 /* limbs */
#define MPFR_EXP_2_THRESHOLD 1023 /* bits */
#define MPFR_EXP_THRESHOLD 10090 /* bits */
#define MPFR_SINCOS_THRESHOLD 23323 /* bits */
#define MPFR_AI_THRESHOLD1 -14098 /* threshold for negative input of mpfr_ai */
#define MPFR_AI_THRESHOLD2 1378
#define MPFR_AI_THRESHOLD3 21450
/* Tuneup completed successfully, took 699 seconds */
