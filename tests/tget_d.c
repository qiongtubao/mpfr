/* Test file for mpfr_get_d

Copyright 1999, 2000, 2001, 2002 Free Software Foundation.

This file is part of the MPFR Library.

The MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version.

The MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the MPFR Library; see the file COPYING.  If not, write to
the Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
MA 02111-1307, USA. */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "mpfr.h"
#include "mpfr-impl.h"
#include "mpfr-test.h"

void check_denorms _PROTO ((void));

void
check_denorms ()
{
  mpfr_t x;
  double d, dd;
  unsigned long int n;

  mpfr_init2 (x, 2);
  
  d = DBL_MIN; /* 2^(-1022) */
  mpfr_set_ui (x, 1, GMP_RNDN);
  mpfr_div_2exp (x, x, 1022, GMP_RNDN); /* 2^(-1022) */
  for (n=0; n<52; n++, d /= 2.0)
    {
      dd = mpfr_get_d (x);
      if (d != dd) /* should be 0 or 2^(-1022-n) */
        {
          fprintf (stderr, "Wrong result for 2^(%ld), ", -1022-n);
          fprintf (stderr, "got %.20e instead of %.20e\n", dd, d);
          exit (1);
        }
      mpfr_div_2exp (x, x, 1, GMP_RNDN);
    }

  mpfr_clear (x);
}

int
main (void)
{

#ifdef HAVE_FENV_H

   mpfr_t half, x, y;
   mp_rnd_t rnd_mode;

   mpfr_init2(half, 2);
   mpfr_set_ui(half, 1, GMP_RNDZ);
   mpfr_div_2ui(half, half, 1, GMP_RNDZ); /* has exponent 0 */

   mpfr_init2(x, 128);
   mpfr_init2(y, 128);

   mpfr_test_init ();

   for (rnd_mode = 0; rnd_mode <= 3; rnd_mode++)
     {
       int i, j, si, sj;
       double di, dj;

       mpfr_set_machine_rnd_mode (rnd_mode);
       for (i = 1, di = 0.25; i < 127; i++, di *= 0.5)
         for (si = 0; si <= 1; si++)
           {
             mpfr_div_2ui (x, half, i, GMP_RNDZ);
             (si ? mpfr_sub : mpfr_add)(x, half, x, GMP_RNDZ);
             /* x = 1/2 +/- 1/2^(1+i) */
             for (j = i+1, dj = di * 0.5; j < 128 && j < i+53; j++, dj *= 0.5)
               for (sj = 0; sj <= 1; sj++)
                 {
                   double c, d, dd;
                   int exp;
                   char *f;

                   mpfr_div_2ui (y, half, j, GMP_RNDZ);
                   (sj ? mpfr_sub : mpfr_add)(y, x, y, GMP_RNDZ);
                   /* y = 1/2 +/- 1/2^(1+i) +/- 1/2^(1+j) */
                   exp = (LONG_RAND() % 47) - 23;
                   mpfr_mul_2si (y, y, exp, GMP_RNDZ);
                   if (mpfr_inexflag_p())
                     {
                       fprintf(stderr, "Error in tget_d: inexact flag for "
                               "(i,si,j,sj,rnd,exp) = (%d,%d,%d,%d,%d,%d)\n",
                               i, si, j, sj, rnd_mode, exp);
                       exit(1);
                     }
                   dd = si != sj ? di - dj : di + dj;
                   d = si ? 0.5 - dd : 0.5 + dd;
                   if ((LONG_RAND() / 1024) & 1)
                     {
                       c = mpfr_get_d2 (y, rnd_mode);
                       f = "mpfr_get_d2";
                     }
                   else
                     {
                       exp = (LONG_RAND() % 47) - 23;
                       c = mpfr_get_d3 (y, exp, rnd_mode);
                       f = "mpfr_get_d3";
                       if (si) /* then real d < 0.5 */
                         d *= sj && i == 1 ? 4 : 2; /* normalize real d */
                     }
                   if (exp > 0)
                     d *= 1 << exp;
                   if (exp < 0)
                     d /= 1 << -exp;
                   if (c != d)
                     {
                       fprintf (stderr, "Error in tget_d (%s) for "
                                "(i,si,j,sj,rnd,exp) = (%d,%d,%d,%d,%d,%d)\n"
                                "got %.25Le instead of %.25Le\n"
                                "Difference: %.19e\n",
                                f, i, si, j, sj, rnd_mode, exp,
                                (long double) c, (long double) d, d - c);
                       exit (1);
                     }
                 }
           }
     }

   mpfr_clear(half);
   mpfr_clear(x);
   mpfr_clear(y);

#endif

   check_denorms ();

   return 0;
}
