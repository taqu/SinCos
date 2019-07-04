#ifndef INC_SINCOS_H_
#define INC_SINCOS_H_
/*
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org>
*/
/**
@file sincos.h
@author t-sakai

USAGE:
Put '#define SINCOS_IMPLEMENTATION' before including this file to create the implementation.
*/
#include <cstdint>

#define SINCOS_IMPLEMENTATION

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

union UnionU32F32
{
    u32 u32_;
    f32 f32_;
};

union UnionU64F64
{
    u64 u64_;
    f64 f64_;
};

template<class T>
T absolute(T val)
{
    return 0<=val? val : -val;
}

template<>
f32 absolute<f32>(f32 val)
{
    UnionU32F32 u;
    u.f32_ = val;
    u.u32_ &= 0x7FFFFFFFU;
    return u.f32_;
}

template<>
f64 absolute<f64>(f64 val)
{
    UnionU64F64 u;
    u.f64_ = val;
    u.u64_ &= 0x7FFFFFFFFFFFFFFFULL;
    return u.f64_;
}

f32 sin_fast(f32 x);
f32 cos_fast(f32 x);
void sincos_fast(f32& dsn, f32& dcs, f32 x);

f64 sin_fast(f64 x);
f64 cos_fast(f64 x);
void sincos_fast(f64& dsn, f64& dcs, f64 x);

#endif //INC_SINCOS_H_

#ifdef SINCOS_IMPLEMENTATION
#define SINCOS_USE_SIMD (0)

#if SINCOS_USE_SIMD
#include <immintrin.h>
#endif

namespace
{
    static const f64 cos_coef[] =
    {
        -5.00000000000000000000e-1,
        4.16666666666666666667e-2,
        -1.38888888888888888889e-3,
        2.48015873015873015873e-5,
        -2.75573192239858906526e-7,
         2.08767569878680989792e-9,
    };

    static const f64 sin_coef[] =
    {
        -1.66666666666666666667e-1,
        8.33333333333333333333e-3,
        -1.98412698412698412698e-4,
        2.75573192239858906526e-6,
        -2.50521083854417187751e-8,
         1.60590438368216145994e-10,
    };

    /**
    @brief fold with cosine's coefficients
    */
    f64 cos_series(f64 x)
    {
#if SINCOS_USE_SIMD
        __m128d x2 = _mm_load_sd(&x);
        x2 = _mm_mul_sd(x2, x2);

        __m128d ret = _mm_load_sd(&cos_coef[4]);
        ret = _mm_fmadd_sd(ret, x2, _mm_load_sd(&cos_coef[3]));
        ret = _mm_fmadd_sd(ret, x2, _mm_load_sd(&cos_coef[2]));
        ret = _mm_fmadd_sd(ret, x2, _mm_load_sd(&cos_coef[1]));
        ret = _mm_fmadd_sd(ret, x2, _mm_load_sd(&cos_coef[0]));
        ret = _mm_mul_sd(ret, x2);
        _mm_store_sd(&x, ret);
        return x + 1.0;
#else
        f64 x2 = x * x;
        f64 ret = cos_coef[5];
        ret = cos_coef[4] + ret * x2;
        ret = cos_coef[3] + ret * x2;
        ret = cos_coef[2] + ret * x2;
        ret = cos_coef[1] + ret * x2;
        ret = cos_coef[0] + ret * x2;
        ret = 1.0 + ret * x2;
        return ret;
#endif
    };


    /**
    @brief fold with cosine's coefficients
    */
    f64 cos_series_x2(f64 x2)
    {
        f64 ret = cos_coef[5];
        ret = cos_coef[4] + ret * x2;
        ret = cos_coef[3] + ret * x2;
        ret = cos_coef[2] + ret * x2;
        ret = cos_coef[1] + ret * x2;
        ret = cos_coef[0] + ret * x2;
        ret = 1.0 + ret * x2;
        return ret;
    };


    /**
    @brief fold with sine's coefficients
    */
    f64 sin_series(f64 x)
    {
#if SINCOS_USE_SIMD
        __m128d tx = _mm_load_sd(&x);
        __m128d x2 = _mm_mul_sd(tx, tx);
        __m128d ret = _mm_load_sd(&sin_coef[4]);
        ret = _mm_fmadd_sd(ret, x2, _mm_load_sd(&sin_coef[3]));
        ret = _mm_fmadd_sd(ret, x2, _mm_load_sd(&sin_coef[2]));
        ret = _mm_fmadd_sd(ret, x2, _mm_load_sd(&sin_coef[1]));
        ret = _mm_fmadd_sd(ret, x2, _mm_load_sd(&sin_coef[0]));
        ret = _mm_add_sd(tx, _mm_mul_sd(_mm_mul_sd(ret, x2), tx));
        _mm_store_sd(&x, ret);
        return x;
#else
        f64 x2 = x * x;
        f64 ret = sin_coef[5];
        ret = sin_coef[4] + ret * x2;
        ret = sin_coef[3] + ret * x2;
        ret = sin_coef[2] + ret * x2;
        ret = sin_coef[1] + ret * x2;
        ret = sin_coef[0] + ret * x2;
        ret = x + ret * x2 * x;
        return ret;
#endif
    };

    /**
    @brief fold with sine's coefficients
    */
    f64 sin_series_x2(f64 x, f64 x2)
    {
        f64 ret = sin_coef[5];
        ret = sin_coef[4] + ret * x2;
        ret = sin_coef[3] + ret * x2;
        ret = sin_coef[2] + ret * x2;
        ret = sin_coef[1] + ret * x2;
        ret = sin_coef[0] + ret * x2;
        ret = x + ret * x2 * x;
        return ret;
    };

    /**
    @brief sin
    */
    f64 sin_core(f64 val)
    {
        const f64 pi2 = 6.28318530717958647692;
        const f64 inv_pi2 = 0.15915494309189533576;

        //sin = -sin
        f64 x = val;
        x = absolute(x);

        //Restrict into [0 pi/2]
        x *= inv_pi2;
        s32 v = (s32)(x);
        x -= v;

        //Branch along with eight areas
        s32 s = (s32)(x*8.0);
        ++s;
        s32 s2 = s >> 1;

        f64 offset = -0.25 * s2;
        x = (x+offset)*pi2;

        bool sign = (s & 0x04U) == 0;
        sign = (val<0.0)? !sign : sign;
        bool isSin = (s2 & 0x01U) == 0;

        f64 ret = isSin? sin_series(x) : cos_series(x);
        return sign? ret : -ret;
    }
}


//
f32 sin_fast(f32 x)
{
    return static_cast<f32>(sin_core(x));
}

//
f32 cos_fast(f32 x)
{
    const f64 pi_2 = 1.57079632679489661923;
    return static_cast<f32>(sin_core(pi_2 + x));
}

//
f64 sin_fast(f64 x)
{
    return sin_core(x);
}

//
f64 cos_fast(f64 x)
{
    const f64 pi_2 = 1.57079632679489661923;
    return sin_core(pi_2 + x);
}

void sincos_fast(f32& dsn, f32& dcs, f32 val)
{
    f64 r0, r1;
    sincos_fast(r0, r1, static_cast<f64>(val));
    dsn = static_cast<f32>(r0);
    dcs = static_cast<f32>(r1);
}

void sincos_fast(f64& dsn, f64& dcs, f64 val)
{
    const f64 pi2 = 6.28318530717958647692;
    const f64 inv_pi2 = 0.15915494309189533576;

    f64 x = val;
    x = absolute(x);

    //Restrict into [0 pi/2]
    x *= inv_pi2;
    s32 v = (s32)(x);
    x -= v;

    //Branch along with eight areas
    s32 s = (s32)(x*8.0);
    ++s;
    s32 s2 = s >> 1;

    f64 offset = -0.25 * s2;
    x = (x+offset)*pi2;

    bool sign = (s & 0x04U) == 0;
    sign = (val<0.0)? !sign : sign;

    bool sign2 = (s<2 || 5<s);

    bool isSin = (s2 & 0x01U) == 0;

    f64 x2 = x*x;

    if(isSin){
        dsn = (sign)? sin_series_x2(x, x2) : -sin_series_x2(x, x2);
        dcs = (sign2)? cos_series_x2(x2) : -cos_series_x2(x2);
    } else{
        dsn = (sign)? cos_series_x2(x2) : -cos_series_x2(x2);
        dcs = (sign2)? sin_series_x2(x, x2) : -sin_series_x2(x, x2);
    }
}

#endif //SINCOS_IMPLEMENTATION
