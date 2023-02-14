/* mpfr_compound_si --- compound(x,n) = (1+x)^n

Copyright 2021-2023 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

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

#define MPFR_NEED_LONGLONG_H /* needed for MPFR_INT_CEIL_LOG2 */
#include "mpfr-impl.h"

/* assuming |(1+x)^n - 1| < 1/4*ulp(1), return correct rounding,
   where s is the sign of n*log2(1+x) */
static int
mpfr_compound_near_one (mpfr_ptr y, int s, mpfr_rnd_t rnd_mode)
{
  mpfr_set_ui (y, 1, rnd_mode); /* exact */
  if (rnd_mode == MPFR_RNDN || rnd_mode == MPFR_RNDF
      || (s > 0 && (rnd_mode == MPFR_RNDZ || rnd_mode == MPFR_RNDD))
      || (s < 0 && (rnd_mode == MPFR_RNDA || rnd_mode == MPFR_RNDU)))
    {
      /* round toward 1 */
      return -s;
    }
  else if (s > 0) /* necessarily RNDA or RNDU */
    {
      /* round toward +Inf */
      mpfr_nextabove (y);
      return +1;
    }
  else /* necessarily s < 0 and RNDZ or RNDD */
    {
      /* round toward 0 */
      mpfr_nextbelow (y);
      return -1;
    }
}

/* put in y the correctly rounded value of (1+x)^n */
int
mpfr_compound_si (mpfr_ptr y, mpfr_srcptr x, long n, mpfr_rnd_t rnd_mode)
{
  int inexact, compared, k, nloop;
  mpfr_t t, u;
  mpfr_exp_t e;
  mpfr_prec_t prec;
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg n=%ld rnd=%d",
      mpfr_get_prec(x), mpfr_log_prec, x, n, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  /* Special cases */
  if (MPFR_IS_SINGULAR (x))
    {
      if (MPFR_IS_INF (x) && MPFR_IS_NEG (x))
        {
          /* compound(-Inf,n) is NaN */
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (n == 0 || MPFR_IS_ZERO (x))
        {
          /* compound(x,0) = 1 for x >= -1 or NaN (the only special value
             of x that is not concerned is -Inf, already handled);
             compound(0,n) = 1 */
          return mpfr_set_ui (y, 1, rnd_mode);
        }
      else if (MPFR_IS_NAN (x))
        {
          /* compound(NaN,n) is NaN, except for n = 0, already handled. */
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x)) /* x = +Inf */
        {
          MPFR_ASSERTD (MPFR_IS_POS (x));
          if (n < 0) /* (1+Inf)^n = +0 for n < 0 */
            MPFR_SET_ZERO (y);
          else /* n > 0: (1+Inf)^n = +Inf */
            MPFR_SET_INF (y);
          MPFR_SET_POS (y);
          MPFR_RET (0); /* exact 0 or infinity */
        }
    }

  /* (1+x)^n = NaN for x < -1 */
  compared = mpfr_cmp_si (x, -1);
  if (compared < 0)
    {
      MPFR_SET_NAN (y);
      MPFR_RET_NAN;
    }

  /* compound(x,0) gives 1 for x >= 1 */
  if (n == 0)
    return mpfr_set_ui (y, 1, rnd_mode);

  if (compared == 0)
    {
      if (n < 0)
        {
          /* compound(-1,n) = +Inf with divide-by-zero exception */
          MPFR_SET_INF (y);
          MPFR_SET_POS (y);
          MPFR_SET_DIVBY0 ();
          MPFR_RET (0);
        }
      else
        {
          /* compound(-1,n) = +0 */
          MPFR_SET_ZERO (y);
          MPFR_SET_POS (y);
          MPFR_RET (0);
        }
    }

  if (n == 1)
    return mpfr_add_ui (y, x, 1, rnd_mode);

  MPFR_SAVE_EXPO_MARK (expo);

  prec = MPFR_PREC(y);
  prec += MPFR_INT_CEIL_LOG2 (prec) + 6;

  mpfr_init2 (t, prec);
  mpfr_init2 (u, prec);

  k = MPFR_INT_CEIL_LOG2(SAFE_ABS (unsigned long, n));  /* thus |n| <= 2^k */

  MPFR_ZIV_INIT (loop, prec);
  for (nloop = 0; ; nloop++)
    {
      unsigned int inex;

      /* we compute (1+x)^n as 2^(n*log2p1(x)) */
      inex = mpfr_log2p1 (t, x, MPFR_RNDN) != 0;
      e = MPFR_GET_EXP(t);
      /* |t - log2(1+x)| <= 1/2*ulp(t) = 2^(e-prec-1) */
      inex |= mpfr_mul_si (t, t, n, MPFR_RNDN) != 0;
      /* |t - n*log2(1+x)| <= 2^(e2-prec-1) + |n|*2^(e-prec-1)
                           <= 2^(e2-prec-1) + 2^(e+k-prec-1) <= 2^(e+k-prec)
                          where |n| <= 2^k, and e2 is the new exponent of t. */
      MPFR_ASSERTD(MPFR_GET_EXP(t) <= e + k);
      e += k;
      /* |t - n*log2(1+x)| <= 2^(e-prec) */
      /* detect overflow */
      mpfr_set_ui (u, 2, MPFR_RNDN);
      mpfr_pow_si (u, u, e - prec, MPFR_RNDN);
      /* now u = 2^(e-prec) */
      mpfr_sub (u, t, u, MPFR_RNDD);
      /* FIXME: An exponent check is not sufficient because the overflow
         and underflow thresholds before rounding are not necessarily
         powers of 2, i.e. one can have an infinite loop (a possible
         solution would be to compare the log2 of the result with the
         log2 of the overflow/underflow threshold, taking the error bound
         into account). Conversely, Patrick Pelissier also reported an
         independent failure about a spurious overflow, which would mean
         that the error analysis is incorrect, as an overflow is generated
         while (1+x)^n < 2^__gmpfr_emax.
         Note: getting the intermediate result with rounding toward 1
         might be a solution to avoid spurious overflow and underflow. */
      /* u <= n*log2(1+x) thus if u >= __gmpfr_emax, then
         (1+x)^n >= 2^__gmpfr_emax and we have overflow */
      if (mpfr_cmp_si (t, __gmpfr_emax) >= 0)
        {
          MPFR_ZIV_FREE (loop);
          mpfr_clear (t);
          mpfr_clear (u);
          MPFR_SAVE_EXPO_FREE (expo);
          return mpfr_overflow (y, rnd_mode, 1);
        }
      /* detect underflow */
      mpfr_set_ui (u, 2, MPFR_RNDN);
      mpfr_pow_si (u, u, e - prec, MPFR_RNDN);
      /* now u = 2^(e-prec) */
      mpfr_add (u, t, u, MPFR_RNDU);
      /* n*log2(1+x) <= u thus if u <= __gmpfr_emin-1, then
         (1+x)^n <= 2^(__gmpfr_emin-1) and we have underflow */
      if (mpfr_cmp_si (u, __gmpfr_emin - 1) <= 0)
        {
          MPFR_ZIV_FREE (loop);
          mpfr_clear (t);
          mpfr_clear (u);
          MPFR_SAVE_EXPO_FREE (expo);
          return mpfr_underflow (y,
                            (rnd_mode == MPFR_RNDN) ? MPFR_RNDZ : rnd_mode, 1);
        }
      /* Detect cases where result is 1 or 1+ulp(1) or 1-1/2*ulp(1):
         |2^t - 1| = |exp(t*log(2)) - 1| <= |t|*log(2) < |t| */
      if (nloop == 0 && MPFR_GET_EXP(t) < - (mpfr_exp_t) MPFR_PREC(y))
        {
          /* since ulp(1) = 2^(1-PREC(y)), we have |t| < 1/4*ulp(1) */
          /* mpfr_compound_near_one must be called in the extended
             exponent range, so that 1 is representable. */
          inexact = mpfr_compound_near_one (y, MPFR_SIGN (t), rnd_mode);
          goto end;
        }
      /* FIXME: mpfr_exp2 could underflow to the smallest positive number
         since MPFR_RNDA is used, and this case will not be detected by
         MPFR_CAN_ROUND (see BUGS). Either fix that, or do early underflow
         detection (which may be necessary). */
      inex |= mpfr_exp2 (t, t, MPFR_RNDA) != 0;
      /* |t - (1+x)^n| <= ulp(t) + |t|*log(2)*2^(e-prec)
                       < 2^(EXP(t)-prec) + 2^(EXP(t)+e-prec) */
      e = (e >= 0) ? e + 1 : 1;
      /* now |t - (1+x)^n| < 2^(EXP(t)+e-prec) */

      if (MPFR_LIKELY (!inex ||
                       MPFR_CAN_ROUND (t, prec - e, MPFR_PREC(y), rnd_mode)))
        break;

      /* Exact cases like compound(0.5,2) = 9/4 must be detected, since
         except for 1+x power of 2, the log2p1 above will be inexact,
         so that in the Ziv test, inexact != 0 and MPFR_CAN_ROUND will
         fail (even for RNDN, as the ternary value cannot be determined),
         yielding an infinite loop.
         For an exact case in precision prec(y), 1+x will necessarily
         be exact in precision prec(y), thus also in prec(t), where
         prec(t) >= prec(y), and we can use mpfr_pow_si under this
         condition (which will also evaluate some non-exact cases). */
      if (mpfr_add_ui (t, x, 1, MPFR_RNDZ) == 0)
        {
          inexact = mpfr_pow_si (y, t, n, rnd_mode);
          goto end;
        }

      MPFR_ZIV_NEXT (loop, prec);
      mpfr_set_prec (t, prec);
    }

  inexact = mpfr_set (y, t, rnd_mode);

 end:
  MPFR_ZIV_FREE (loop);
  mpfr_clear (t);
  mpfr_clear (u);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd_mode);
}
