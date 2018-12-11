#pragma once

#include "vec.h"

/* epsilon surrounding for near zero values */
const float EQN_EPS = 1e-9;

inline __device__ bool isZero(float x) {
    return ((x) > -EQN_EPS && (x) < EQN_EPS);
}

inline __device__ void solver2(const float coef2[3], float (&sol2)[2], int &n2) {
    // normal form: x^2 + px + q = 0
    float p = coef2[1] / (2.f * coef2[2]);
    float q = coef2[0] / coef2[2];

    float D = p * p - q;

    if(isZero(D)) {
        n2 = 1;
        sol2[0] = -p;
        return;
    } 
    else if (D < 0) {
        n2 = 0;
        return;
    } 
    else {
        sol2[0] = sqrt(D) - p;
        sol2[1] = - sqrt(D) - p;
        n2 = 2;
    }
}

inline __device__ void solver3(const float coef3[4], float (&sol3)[3], int &n3) {
    // normal form: x^3 + Ax^2 + Bx + C = 0

}

// solves c[0] + c[1]*x + c[2]*x^2 + c[3]*x^3 + c[4]*x^4 = 0
inline __device__ void solver4(const float coef4[5], float (&sol4)[4], int &n4) {
    // normal form: x^4 + Ax^3 + Bx^2 + Cx + D = 0
    float A = coef4[3] / coef4[4];
    float B = coef4[2] / coef4[4];
    float C = coef4[1] / coef4[4];
    float D = coef4[0] / coef4[4];

    // substitute x = y - A/4 to eliminate cubic term:	
    // x^4 + px^2 + qx + r = 0
    float sq_A = A * A;
    float p = -3.f / 8 * sq_A + B;
    float q = 1.f / 8 * sq_A * A - 1.f / 2 * A * B + C;
    float r = - 3.f / 256 * sq_A * sq_A + 1.f / 16 * sq_A * B - 1.f / 4 * A * B * C + D;

    if(isZero(r)) {
        float sol3[3];

        // no absolute term: y(y^3 + py + q) = 0
        float coef3[4] = {q, p, 0.f, 1.f};
        int n3;
        solver3(coef3, sol3, n3);

        sol4[0] = sol3[0];
        sol4[1] = sol3[1];
        sol4[2] = sol3[2];
        sol4[3] = 0.f;

        n4 = n3 + 1;
    }
    else {
        float sol3[3];

        // solve the resolvent cubic
        float coef3[4] = {1.f / 2.f * r * p - 1.f / 8.f * q * q, 
                          -r, 
                          - 1.f / 2.f * p, 
                          1.f};
        int n3;
        solver3(coef3, sol3, n3);

        // take the one real solution...
        float z = sol3[0];
        
        // ...to build two quadric equations
        float u = z * z - r;
        float v = 2 * z - p;

        if(isZero(u))
            u = 0;
        else if(u > 0)
            u = sqrt(u);
        else{
            n4 = 0; // no solution
            return;
        }

        if(isZero(v))
            v = 0;
        else if(v > 0)
            v = sqrt(v);
        else {
            n4 = 0; // no solution
            return;
        }

        int n21;
        float sol21[2];
        float coef21[3] = {z - u, (q < 0 ? -v : v), 1};
        solver2(coef21, sol21, n21);

        int n22;
        float sol22[2];
        float coef22[3] = {z + u, q < 0 ? v : -v, 1};
        solver2(coef22, sol22, n22);

        sol4[0] = sol21[0];
        sol4[1] = sol21[1];
        sol4[2] = sol22[0];
        sol4[3] = sol22[1];
        n4 = n21 + n22;
    }

    // resubstitute 
    float sub = 1.f / 4.f * A;
    for(int i = 0; i < n4; i++)
        sol4[i] -= sub;
}