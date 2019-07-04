#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#define SINCOS_IMPLEMENTATION
#include "sincos.h"

static constexpr f32 PI = static_cast<f32>(3.14159265358979323846);
static constexpr f32 PI2 = static_cast<f32>(6.28318530717958647692);

static constexpr f64 PI_64 = static_cast<f64>(3.14159265358979323846);
static constexpr f64 PI2_64 = static_cast<f64>(6.28318530717958647692);

int main(int /*argc*/, char** /*argv*/)
{
    s32 NUM_SAMPLES = 1024 * 1024;
    s32 total = NUM_SAMPLES * 2 + 1;

    {//double precision
        printf("double precision\n----------------\n");
        f64* data = (f64*)malloc(sizeof(f64) * total * 3);
        f64* values = data;
        f64* results0 = values + total;
        f64* results1 = results0 + total;

        for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
            f64 x = PI2_64 * 4.0 / NUM_SAMPLES * i;
            values[i + NUM_SAMPLES] = x;
        }

        {
            auto start0 = std::chrono::high_resolution_clock::now();
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                results0[i + NUM_SAMPLES] = sin_fast(values[i + NUM_SAMPLES]);
            }
            std::chrono::nanoseconds duration0 = std::chrono::high_resolution_clock::now() - start0;

            auto start1 = std::chrono::high_resolution_clock::now();
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                results1[i + NUM_SAMPLES] = sin(values[i + NUM_SAMPLES]);
            }
            std::chrono::nanoseconds duration1 = std::chrono::high_resolution_clock::now() - start1;

            f64 error = 0.0;
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                f64 e = absolute(results0[i + NUM_SAMPLES] - results1[i + NUM_SAMPLES]);
                error = error<e? e : error;
            }
            printf("sin max error %1.16lf\n", error);
            printf("sin time fast:%lld, math:%lld\n", duration0.count(), duration1.count());
        }

        {
            auto start0 = std::chrono::high_resolution_clock::now();
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                results0[i + NUM_SAMPLES] = cos_fast(values[i + NUM_SAMPLES]);
            }
            std::chrono::nanoseconds duration0 = std::chrono::high_resolution_clock::now() - start0;

            auto start1 = std::chrono::high_resolution_clock::now();
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                results1[i + NUM_SAMPLES] = cos(values[i + NUM_SAMPLES]);
            }
            std::chrono::nanoseconds duration1 = std::chrono::high_resolution_clock::now() - start1;

            f64 error = 0.0;
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                f64 e = absolute(results0[i + NUM_SAMPLES] - results1[i + NUM_SAMPLES]);
                error = error<e? e : error;
            }
            printf("cos max error %1.16lf\n", error);
            printf("cos time fast:%lld, math:%lld\n", duration0.count(), duration1.count());
        }
        free(data);
    }
    {//single precision
        printf("single precision\n----------------\n");
        f32* data = (f32*)malloc(sizeof(f32) * total * 3);
        f32* values = data;
        f32* results0 = values + total;
        f32* results1 = results0 + total;

        for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
            f32 x = PI2 * 16.0f / NUM_SAMPLES * i;
            values[i + NUM_SAMPLES] = x;
        }

        {
            auto start0 = std::chrono::high_resolution_clock::now();
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                results0[i + NUM_SAMPLES] = sin_fast(values[i + NUM_SAMPLES]);
            }
            std::chrono::nanoseconds duration0 = std::chrono::high_resolution_clock::now() - start0;

            auto start1 = std::chrono::high_resolution_clock::now();
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                results1[i + NUM_SAMPLES] = sinf(values[i + NUM_SAMPLES]);
            }
            std::chrono::nanoseconds duration1 = std::chrono::high_resolution_clock::now() - start1;

            f64 error = 0.0;
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                f64 e = absolute(results0[i + NUM_SAMPLES] - results1[i + NUM_SAMPLES]);
                error = error<e? e : error;
            }
            printf("sin max error %1.16lf\n", error);
            printf("sin time fast:%lld, math:%lld\n", duration0.count(), duration1.count());
        }

        {
            auto start0 = std::chrono::high_resolution_clock::now();
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                results0[i + NUM_SAMPLES] = cos_fast(values[i + NUM_SAMPLES]);
            }
            std::chrono::nanoseconds duration0 = std::chrono::high_resolution_clock::now() - start0;

            auto start1 = std::chrono::high_resolution_clock::now();
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                results1[i + NUM_SAMPLES] = cosf(values[i + NUM_SAMPLES]);
            }
            std::chrono::nanoseconds duration1 = std::chrono::high_resolution_clock::now() - start1;

            f64 error = 0.0;
            for(int i = -NUM_SAMPLES; i <= NUM_SAMPLES; ++i){
                f64 e = absolute(results0[i + NUM_SAMPLES] - results1[i + NUM_SAMPLES]);
                error = error<e? e : error;
            }
            printf("cos max error %1.16lf\n", error);
            printf("cos time fast:%lld, math:%lld\n", duration0.count(), duration1.count());
        }
        free(data);
    }
    return 0;
}
