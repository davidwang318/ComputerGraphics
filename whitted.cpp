/**
This program is originally implemented by:

Copyright (C) 2012  www.scratchapixel.com
including the basic algorithm of ray tracing

and modified by:

Lih-Narn Wang
Master of Engineering in Robotics
University of Maryland, College Park
Email: ytcdavid@terpmail.umd.edu

including:
Anti-aliasing
Box object
Cylinder object
Procedure texutures

-------------------------------------------

Running command:
g++ whitted.cpp
./a.out

the result will be "out.ppm"
**/

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <utility>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include <limits>
#include <cstring>
#include <algorithm>

#include "perlin.cpp"

#define PI 3.14159265

const float kInfinity = std::numeric_limits<float>::max();

class Vec3f {
public:
    Vec3f() : x(0), y(0), z(0) {}
    Vec3f(float xx) : x(xx), y(xx), z(xx) {}
    Vec3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
    Vec3f operator * (const float &r) const { return Vec3f(x * r, y * r, z * r); }
    Vec3f operator * (const Vec3f &v) const { return Vec3f(x * v.x, y * v.y, z * v.z); }
    Vec3f operator - (const Vec3f &v) const { return Vec3f(x - v.x, y - v.y, z - v.z); }
    Vec3f operator + (const Vec3f &v) const { return Vec3f(x + v.x, y + v.y, z + v.z); }
    Vec3f operator - () const { return Vec3f(-x, -y, -z); }
    Vec3f operator / (int n) { return Vec3f(x/n, y/n, z/n); }
    Vec3f& operator += (const Vec3f &v) { x += v.x, y += v.y, z += v.z; return *this; }
    friend Vec3f operator * (const float &r, const Vec3f &v)
    { return Vec3f(v.x * r, v.y * r, v.z * r); }
    friend std::ostream & operator << (std::ostream &os, const Vec3f &v)
    { return os << v.x << ", " << v.y << ", " << v.z; }
    float x, y, z;
};

class Vec2f
{
public:
    Vec2f() : x(0), y(0) {}
    Vec2f(float xx) : x(xx), y(xx) {}
    Vec2f(float xx, float yy) : x(xx), y(yy) {}
    Vec2f operator * (const float &r) const { return Vec2f(x * r, y * r); }
    Vec2f operator + (const Vec2f &v) const { return Vec2f(x + v.x, y + v.y); }
    float x, y;
};

Vec3f normalize(const Vec3f &v)
{
    float mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (mag2 > 0) {
        float invMag = 1 / sqrtf(mag2);
        return Vec3f(v.x * invMag, v.y * invMag, v.z * invMag);
    }

    return v;
}

inline
float dotProduct(const Vec3f &a, const Vec3f &b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3f crossProduct(const Vec3f &a, const Vec3f &b)
{
    return Vec3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline
float clamp(const float &lo, const float &hi, const float &v)
{ return std::max(lo, std::min(hi, v)); }

inline
float deg2rad(const float &deg)
{ return deg * M_PI / 180; }

inline
Vec3f mix(const Vec3f &a, const Vec3f& b, const float &mixValue)
{ return a * (1 - mixValue) + b * mixValue; }

struct Options
{
    uint32_t width;
    uint32_t height;
    float fov;
    float imageAspectRatio;
    uint8_t maxDepth;
    Vec3f backgroundColor;
    float bias;
};

class Light
{
public:
    Light(const Vec3f &p, const Vec3f &i) : position(p), intensity(i) {}
    Vec3f position;
    Vec3f intensity;
};

enum MaterialType { DIFFUSE_AND_GLOSSY, REFLECTION_AND_REFRACTION, REFLECTION };

bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0) return false;
    else if (discr == 0) x0 = x1 = - 0.5 * b / a;
    else {
        float q = (b > 0) ?
            -0.5 * (b + sqrt(discr)) :
            -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) std::swap(x0, x1);
    return true;
}

bool rayTriangleIntersect(
    const Vec3f &v0, const Vec3f &v1, const Vec3f &v2,
    const Vec3f &orig, const Vec3f &dir,
    float &tnear, float &u, float &v)
{
    Vec3f edge1 = v1 - v0;
    Vec3f edge2 = v2 - v0;
    Vec3f pvec = crossProduct(dir, edge2);
    float det = dotProduct(edge1, pvec);
    if (det == 0 || det < 0) return false;

    Vec3f tvec = orig - v0;
    u = dotProduct(tvec, pvec);
    if (u < 0 || u > det) return false;

    Vec3f qvec = crossProduct(tvec, edge1);
    v = dotProduct(dir, qvec);
    if (v < 0 || u + v > det) return false;

    float invDet = 1 / det;
    
    tnear = dotProduct(edge2, qvec) * invDet;
    u *= invDet;
    v *= invDet;

    return true;
}

class Object
{
 public:
    Object(float ior_=1.3, float kd_=0.8, float ks_=0.2, 
        Vec3f diffuseColor_=Vec3f(0.2), float specularExponent_=25) :
        materialType(DIFFUSE_AND_GLOSSY),
        ior(ior_), Kd(kd_), Ks(ks_), diffuseColor(diffuseColor_), specularExponent(specularExponent_) {}
    virtual ~Object() {}
    virtual bool intersect(const Vec3f &, const Vec3f &, float &, uint32_t &, Vec2f &) const = 0;
    virtual void getSurfaceProperties(const Vec3f &, const Vec3f &, const uint32_t &, const Vec2f &, Vec3f &, Vec2f &) const = 0;
    virtual Vec3f evalDiffuseColor(const Vec2f &) const { return diffuseColor; }
    virtual bool intersectPlane(float& t, const Vec3f &n, const float &d, const Vec3f& orig, const Vec3f &dir) const{
        float deno = dotProduct(n, dir);
        if (deno == 0) return false;
        t = (d - dotProduct(n, orig)) / deno;
        return true; 
    }

    float chessBoard(const Vec2f &st) const {
        float scale = 3;
        return (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
    }

    float perlin(const Vec2f &st, int freq) const {
        return Perlin_Get2d(st.x, st.y, freq, 1);
    }

    float stripe(const Vec2f &st, int freq) const {
        return sin(pow((st.x-0.5)*(st.x-0.5)+(st.y-0.5)*(st.y-0.5), 0.5) * freq * 2 * PI);
    }

    float perlinChessBoard(const Vec2f& st) const {
        Vec2f stRev;
        stRev.x = 1 - st.x;
        stRev.y = 1 - st.y;
        Vec2f st2;
        st2.x = perlin(st, 10);
        st2.y = perlin(stRev, 10);
        return chessBoard(st2);
    }

    float perlinStripe(const Vec2f& st) const {
        Vec2f stRev;
        stRev.x = 1 - st.x;
        stRev.y = 1 - st.y;
        Vec2f st2;
        st2.x = perlin(st, 10);
        st2.y = perlin(stRev, 10);
        return stripe(st2, 5);
    }
    // material properties
    MaterialType materialType;
    float ior;
    float Kd, Ks;
    Vec3f diffuseColor;
    float specularExponent;
};

class Sphere : public Object{
public:
    Sphere(const Vec3f &c, const float &r, float ior_=1.3, float kd_=0.8, float ks_=0.2, 
        Vec3f diffuseColor_=Vec3f(0.2), float specularExponent_=25): center(c), radius(r), radius2(r * r)
    , Object(ior_, kd_, ks_, diffuseColor_, specularExponent_){}
    bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear, uint32_t &index, Vec2f &uv) const
    {
        // analytic solution
        Vec3f L = orig - center;
        float a = dotProduct(dir, dir);
        float b = 2 * dotProduct(dir, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return false;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        tnear = t0;

        return true;
    }
    
    void getSurfaceProperties(const Vec3f &P, const Vec3f &I, const uint32_t &index, const Vec2f &uv, Vec3f &N, Vec2f &st) const
    { 
    	N = normalize(P - center);
    	st.x = 0.5 + atan2(N.x, N.z) / 2.0 / PI;
    	st.y = 0.5 - asin(N.y) / PI;
    	return;
    }

    Vec3f evalDiffuseColor(const Vec2f &st) const
    {
        Vec2f stx = Vec2f(st.x, st.x);
        Vec2f sty = Vec2f(st.y, st.y);
        auto pattern1 = perlin(stx, 10);
        auto pattern2 = perlin(sty, 10);

        return (mix(Vec3f(0.3, 0.9, 0.5), Vec3f(0., 1., 0.), sin(pattern1*10*PI)) 
        + mix(Vec3f(0.733, 0.15, 0.24), Vec3f(0.47, 0.84, 0.93), cos(pattern2*10*PI))) / 2.0;
    }

    Vec3f center;
    float radius, radius2;
};

class MeshTriangle : public Object
{
public:
    MeshTriangle(
        const Vec3f *verts,
        const uint32_t *vertsIndex,
        const uint32_t &numTris,
        const Vec2f *st)
    {
        uint32_t maxIndex = 0;
        for (uint32_t i = 0; i < numTris * 3; ++i)
            if (vertsIndex[i] > maxIndex) maxIndex = vertsIndex[i];
        maxIndex += 1;
        vertices = std::unique_ptr<Vec3f[]>(new Vec3f[maxIndex]);
        memcpy(vertices.get(), verts, sizeof(Vec3f) * maxIndex);
        vertexIndex = std::unique_ptr<uint32_t[]>(new uint32_t[numTris * 3]);
        memcpy(vertexIndex.get(), vertsIndex, sizeof(uint32_t) * numTris * 3);
        numTriangles = numTris;
        stCoordinates = std::unique_ptr<Vec2f[]>(new Vec2f[maxIndex]);
        memcpy(stCoordinates.get(), st, sizeof(Vec2f) * maxIndex);
    }

    bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear, uint32_t &index, Vec2f &uv) const
    {
        bool intersect = false;
        for (uint32_t k = 0; k < numTriangles; ++k) {
            const Vec3f & v0 = vertices[vertexIndex[k * 3]];
            const Vec3f & v1 = vertices[vertexIndex[k * 3 + 1]];
            const Vec3f & v2 = vertices[vertexIndex[k * 3 + 2]];
            float t, u, v;
            if (rayTriangleIntersect(v0, v1, v2, orig, dir, t, u, v) && t < tnear) {
                tnear = t;
                uv.x = u;
                uv.y = v;
                index = k;
                intersect |= true;
            }
        }

        return intersect;
    }

    void getSurfaceProperties(const Vec3f &P, const Vec3f &I, const uint32_t &index, const Vec2f &uv, Vec3f &N, Vec2f &st) const
    {
        const Vec3f &v0 = vertices[vertexIndex[index * 3]];
        const Vec3f &v1 = vertices[vertexIndex[index * 3 + 1]];
        const Vec3f &v2 = vertices[vertexIndex[index * 3 + 2]];
        Vec3f e0 = normalize(v1 - v0);
        Vec3f e1 = normalize(v2 - v1);
        N = normalize(crossProduct(e0, e1));
        const Vec2f &st0 = stCoordinates[vertexIndex[index * 3]];
        const Vec2f &st1 = stCoordinates[vertexIndex[index * 3 + 1]];
        const Vec2f &st2 = stCoordinates[vertexIndex[index * 3 + 2]];
        st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y;
    }

    Vec3f evalDiffuseColor(const Vec2f &st) const
    { return mix(Vec3f(0., 0., 0.), Vec3f(1., 1., 1.), perlinStripe(st));}

    std::unique_ptr<Vec3f[]> vertices;
    uint32_t numTriangles;
    std::unique_ptr<uint32_t[]> vertexIndex;
    std::unique_ptr<Vec2f[]> stCoordinates;
};

class Box : public Object
{
public:
    // Plane equatin: n*x = d
    float dSlabX1, dSlabX2, dSlabY1, dSlabY2, dSlabZ1, dSlabZ2;
    Vec3f centerPts, xAxis, yAxis, zAxis;
    float l;

    Box(const Vec3f centerPts, const float length,
        const Vec3f xAxis, const Vec3f yAxis, const Vec3f zAxis):
        centerPts(centerPts), xAxis(xAxis), yAxis(yAxis), zAxis(zAxis){            
        l = length / 2;
        dSlabX1 = dotProduct(centerPts - xAxis*l, xAxis);
        dSlabX2 = dotProduct(centerPts + xAxis*l, xAxis);
        dSlabY1 = dotProduct(centerPts - yAxis*l, yAxis);
        dSlabY2 = dotProduct(centerPts + yAxis*l, yAxis);
        dSlabZ1 = dotProduct(centerPts - zAxis*l, zAxis);
        dSlabZ2 = dotProduct(centerPts + zAxis*l, zAxis);
    }

    bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear, uint32_t &index, Vec2f &uv) const{
        float xDist = axisDistance(orig, xAxis), yDist = axisDistance(orig, yAxis), zDist = axisDistance(orig, zAxis);
        // Special Case: Parallel to the surface and outside the box
        if ((dotProduct(xAxis, dir) == 0 && xDist > l) ||
            (dotProduct(yAxis, dir) == 0 && yDist > l) ||
            (dotProduct(zAxis, dir) == 0 && zDist > l))
            {return false;}
        // General Case:
        float xEnter{std::numeric_limits<float>::min()}, yEnter{xEnter}, zEnter{xEnter};
        float xExit{std::numeric_limits<float>::max()}, yExit{xExit}, zExit{xExit};
        float tEnter, tExit;
        if(intersectPlane(xEnter, xAxis, dSlabX1, orig, dir) && intersectPlane(xExit, xAxis, dSlabX2, orig, dir)){
            if (xEnter > xExit) std::swap(xEnter, xExit);
        }
        if(intersectPlane(yEnter, yAxis, dSlabY1, orig, dir) && intersectPlane(yExit, yAxis, dSlabY2, orig, dir)){
            if (yEnter > yExit) std::swap(yEnter, yExit);
        }
        if(intersectPlane(zEnter, zAxis, dSlabZ1, orig, dir) && intersectPlane(zExit, zAxis, dSlabZ2, orig, dir)){
            if (zEnter > zExit) std::swap(zEnter, zExit);
        }
        tEnter = std::max(std::max(xEnter, yEnter), zEnter);
        tExit = std::min(std::min(xExit, yExit), zExit);
        if(tEnter > tExit || tEnter < 0) {return false;}
        tnear = tEnter;
             
        return true;
    }

    void getSurfaceProperties(const Vec3f &P, const Vec3f &I, const uint32_t &index, const Vec2f &uv, Vec3f &N, Vec2f &st) const
    {
        Vec3f diff = P - centerPts;
        Vec3f diffNormal = normalize(diff);
        float xDot = dotProduct(diff, xAxis), yDot = dotProduct(diff, yAxis), zDot = dotProduct(diff, zAxis);
        float stx {0}, sty {0};
        // Y-Z plane
        if (std::abs(std::abs(xDot) - l) <= std::abs(std::abs(yDot) - l)
            && std::abs(std::abs(xDot) - l) <= std::abs(std::abs(zDot) - l)){
            N = xDot > 0? xAxis : -xAxis;
            stx = yDot / l;
            sty = zDot / l;
        }
        // X-Z plane
        else if(std::abs(std::abs(yDot) - l) <= std::abs(std::abs(xDot) - l) 
            && std::abs(std::abs(yDot) - l) <= std::abs(std::abs(zDot) - l)){
            N = yDot > 0? yAxis : -yAxis;
            stx = xDot / l;
            sty = zDot / l;
        }
        // X-Y plane
        else {
            N = zDot > 0? zAxis : -zAxis;
            stx = xDot / l;
            sty = yDot / l;
        }
        st.x = (stx + 1) / 2.0;
        st.y = (sty + 1) / 2.0;
    }

    Vec3f evalDiffuseColor(const Vec2f &st) const
    {
        float pattern1 = chessBoard(st);
        float pattern2 = perlin(st, 10);

        return (mix(Vec3f(0.25, 0.25, 0.), Vec3f(0., 0.25, 0.25), pattern1) 
              + mix(Vec3f(0.75, 0.25, 0.75), Vec3f(0.83, 0.92, 0.3), pattern2)) / 2.0;
    }
private:

    float axisDistance(const Vec3f &orig, const Vec3f &axis) const{
        return std::abs(dotProduct((orig - centerPts), axis));
    }
};

class Cylinder: public Object
{
public:
    // Attributes:
    Vec3f c1, c2, v, vNeg;
    float r, dUp, dDown, h;
    // Constructer:
    Cylinder(const Vec3f& c1, const Vec3f& c2, float& r):c1(c1), c2(c2), r(r)  
    {
        auto diff = c2 - c1;
        v = normalize(diff);
        vNeg = -v;
        h = pow(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z, 0.5);
        dUp = dotProduct(c2, v);
        dDown = dotProduct(c1, -v);
    }
    bool intersect(const Vec3f &orig, const Vec3f &dir, float &tnear, uint32_t &index, Vec2f &uv) const{
        std::vector<float> tCandidate;
        // Intersection with the infinite cylinder
        float tCylinderMin, tCylinderMax;
        Vec3f pDiff = orig - c1;
        Vec3f aVec = dir - dotProduct(dir, v)*v;
        Vec3f cVec = pDiff - dotProduct(pDiff, v)*v;
        float A = aVec.x*aVec.x + aVec.y*aVec.y + aVec.z*aVec.z;
        float B = 2 * dotProduct(dir - dotProduct(dir, v)*v, pDiff - dotProduct(pDiff, v)*v);
        float C = cVec.x*cVec.x + cVec.y*cVec.y + cVec.z*cVec.z - r*r;
        bool cylinderIntersection = solveQuadratic(A, B, C, tCylinderMin, tCylinderMax);
        if (!cylinderIntersection || tCylinderMax < 0){ 
            return false;
        }
        else{
            Vec3f hitPoint = orig + dir * tCylinderMin;
            if(tCylinderMin > 0 && dotProduct(v, hitPoint - c1) > 0 && dotProduct(v, hitPoint - c2) < 0){
                tCandidate.push_back(tCylinderMin);
            }
            hitPoint = orig + dir * tCylinderMax;
            if(dotProduct(v, hitPoint - c1) > 0 && dotProduct(v, hitPoint - c2) < 0){
                tCandidate.push_back(tCylinderMax);
            }
        }
        // Intersection with the up & down cap
        float tUpper = -1, tDown = -1;
        bool upperIntersection = intersectPlane(tUpper, v, dUp, orig, dir);
        bool downIntersection = intersectPlane(tDown, vNeg, dDown, orig, dir);
        if (upperIntersection && tUpper >= 0){
            auto d = (orig + tUpper * dir) - c2;
            if (d.x*d.x + d.y*d.y + d.z*d.z <= r*r){
                tCandidate.push_back(tUpper);
            }
        } 
        if (downIntersection && tDown >= 0){
            auto d = (orig + tDown * dir) - c1;
            if (d.x*d.x + d.y*d.y + d.z*d.z <= r*r){
                tCandidate.push_back(tDown);
            }
        }
        // Return the smalles value
        if (tCandidate.empty()) return false;
        tnear = tCandidate.back(); tCandidate.pop_back();
        for(const auto & t : tCandidate)
            tnear = t < tnear? t : tnear;
        return true;
    }
    void getSurfaceProperties(const Vec3f &P, const Vec3f &I, const uint32_t &index, const Vec2f &uv, Vec3f &N, Vec2f &st) const{
        float error = 0.0000001;
        float vDistance = std::abs(dotProduct(P - c1, v));
        // Points on the down cap
        if (vDistance < error) {
            N = -v;
            circleUVcalculation(P, c1, st); 
            return;
        }
        // Points on the up cap
        if (std::abs(vDistance - h) < error) {
            N = v;
            circleUVcalculation(P, c2, st); 
            return;
        }
        // Points on the cylinder
        auto diff = P - c1;
        N = normalize(diff - dotProduct(diff, v)*v);
        st.x = 0.5 + atan2(N.x, N.z) / 2.0 / PI;
        st.y = vDistance;
        return;
    }

    Vec3f evalDiffuseColor(const Vec2f &st) const
    {
        auto pattern1 = perlin(st, 10);
        auto pattern2 = sin(pattern1 * 10 * PI);
        auto pattern3 = cos(pattern2 * 10 * PI);
        return mix(Vec3f(1., 1., 0.), Vec3f(0., 1., 1.), (1 + pattern3) / 2.0);
    }
private:
    void circleUVcalculation(const Vec3f &P, const Vec3f &center, Vec2f&st) const {
        // Vec3f coor = normalize(P - center);
        // st.x = (0.5 * (pow(2 + 2 * coor.x * pow(2, 0.5) + coor.x * coor.x - coor.z * coor.z, 0.5) -
        //                pow(2 - 2 * coor.x * pow(2, 0.5) + coor.x * coor.x - coor.z * coor.z, 0.5)) + 1) / 2.0;
        // st.y = (0.5 * (pow(2 + 2 * coor.z * pow(2, 0.5) - coor.x * coor.x + coor.z * coor.z, 0.5) -
        //                pow(2 - 2 * coor.z * pow(2, 0.5) - coor.x * coor.x + coor.z * coor.z, 0.5)) + 1) / 2.0;
        st.x = 0;
        st.y = 0;
    }
};

// [comment]
// Compute reflection direction
// [/comment]
Vec3f reflect(const Vec3f &I, const Vec3f &N)
{
    return I - 2 * dotProduct(I, N) * N;
}

// [comment]
// Compute refraction direction using Snell's law
//
// We need to handle with care the two possible situations:
//
//    - When the ray is inside the object
//
//    - When the ray is outside.
//
// If the ray is outside, you need to make cosi positive cosi = -N.I
//
// If the ray is inside, you need to invert the refractive indices and negate the normal N
// [/comment]
Vec3f refract(const Vec3f &I, const Vec3f &N, const float &ior)
{
    float cosi = clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;
    Vec3f n = N;
    if (cosi < 0) { cosi = -cosi; } else { std::swap(etai, etat); n= -N; }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}

// [comment]
// Compute Fresnel equation
//
// \param I is the incident view direction
//
// \param N is the normal at the intersection point
//
// \param ior is the mateural refractive index
//
// \param[out] kr is the amount of light reflected
// [/comment]
void fresnel(const Vec3f &I, const Vec3f &N, const float &ior, float &kr)
{
    float cosi = clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;
    if (cosi > 0) {  std::swap(etai, etat); }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        kr = 1;
    }
    else {
        float cost = sqrtf(std::max(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);
        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        kr = (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
}

// [comment]
// Returns true if the ray intersects an object, false otherwise.
//
// \param orig is the ray origin
//
// \param dir is the ray direction
//
// \param objects is the list of objects the scene contains
//
// \param[out] tNear contains the distance to the cloesest intersected object.
//
// \param[out] index stores the index of the intersect triangle if the interesected object is a mesh.
//
// \param[out] uv stores the u and v barycentric coordinates of the intersected point
//
// \param[out] *hitObject stores the pointer to the intersected object (used to retrieve material information, etc.)
//
// \param isShadowRay is it a shadow ray. We can return from the function sooner as soon as we have found a hit.
// [/comment]
bool trace(
    const Vec3f &orig, const Vec3f &dir,
    const std::vector<std::unique_ptr<Object>> &objects,
    float &tNear, uint32_t &index, Vec2f &uv, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vec2f uvK;
        if (objects[k]->intersect(orig, dir, tNearK, indexK, uvK) && tNearK < tNear) {
            *hitObject = objects[k].get();
            tNear = tNearK;
            index = indexK;
            uv = uvK;
        }
    }

    return (*hitObject != nullptr);
}

// [comment]
// Implementation of the Whitted-syle light transport algorithm (E [S*] (D|G) L)
//
// This function is the function that compute the color at the intersection point
// of a ray defined by a position and a direction. Note that thus function is recursive (it calls itself).
//
// If the material of the intersected object is either reflective or reflective and refractive,
// then we compute the reflection/refracton direction and cast two new rays into the scene
// by calling the castRay() function recursively. When the surface is transparent, we mix
// the reflection and refraction color using the result of the fresnel equations (it computes
// the amount of reflection and refractin depending on the surface normal, incident view direction
// and surface refractive index).
//
// If the surface is duffuse/glossy we use the Phong illumation model to compute the color
// at the intersection point.

// get the hitpoint, take a vector from the sphere' 
// s center to the hitpoint 
// then take the normal 
// [/comment]
Vec3f castRay(
    const Vec3f &orig, const Vec3f &dir,
    const std::vector<std::unique_ptr<Object>> &objects,
    const std::vector<std::unique_ptr<Light>> &lights,
    const Options &options,
    uint32_t depth,
    bool test = false)
{
    if (depth > options.maxDepth) {
        return options.backgroundColor;
    }
    
    Vec3f hitColor = options.backgroundColor;
    float tnear = kInfinity;
    Vec2f uv;
    uint32_t index = 0;
    Object *hitObject = nullptr;
    if (trace(orig, dir, objects, tnear, index, uv, &hitObject)) {
        Vec3f hitPoint = orig + dir * tnear;
        Vec3f N; // normal
        Vec2f st; // st coordinates
        hitObject->getSurfaceProperties(hitPoint, dir, index, uv, N, st);
        Vec3f tmp = hitPoint;
        switch (hitObject->materialType) {
            case REFLECTION_AND_REFRACTION:
            {
                Vec3f reflectionDirection = normalize(reflect(dir, N));
                Vec3f refractionDirection = normalize(refract(dir, N, hitObject->ior));
                Vec3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                    hitPoint - N * options.bias :
                    hitPoint + N * options.bias;
                Vec3f refractionRayOrig = (dotProduct(refractionDirection, N) < 0) ?
                    hitPoint - N * options.bias :
                    hitPoint + N * options.bias;
                Vec3f reflectionColor = castRay(reflectionRayOrig, reflectionDirection, objects, lights, options, depth + 1, 1);
                Vec3f refractionColor = castRay(refractionRayOrig, refractionDirection, objects, lights, options, depth + 1, 1);
                float kr;
                fresnel(dir, N, hitObject->ior, kr);
                hitColor = reflectionColor * kr + refractionColor * (1 - kr);
                break;
            }
            case REFLECTION:
            {
                float kr;
                fresnel(dir, N, hitObject->ior, kr);
                Vec3f reflectionDirection = reflect(dir, N);
                Vec3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                    hitPoint + N * options.bias :
                    hitPoint - N * options.bias;
                hitColor = castRay(reflectionRayOrig, reflectionDirection, objects, lights, options, depth + 1) * kr;
                break;
            }
            default:
            {
                // [comment]
                // We use the Phong illumation model int the default case. The phong model
                // is composed of a diffuse and a specular reflection component.
                // [/comment]
                Vec3f lightAmt = 0, specularColor = 0;
                Vec3f shadowPointOrig = (dotProduct(dir, N) < 0) ?
                    hitPoint + N * options.bias :
                    hitPoint - N * options.bias;
                // [comment]
                // Loop over all lights in the scene and sum their contribution up
                // We also apply the lambert cosine law here though we haven't explained yet what this means.
                // [/comment]
                for (uint32_t i = 0; i < lights.size(); ++i) {
                    Vec3f lightDir = lights[i]->position - hitPoint;
                    // square of the distance between hitPoint and the light
                    float lightDistance2 = dotProduct(lightDir, lightDir);
                    lightDir = normalize(lightDir);
                    float LdotN = std::max(0.f, dotProduct(lightDir, N));
                    Object *shadowHitObject = nullptr;
                    float tNearShadow = kInfinity;
                    // is the point in shadow, and is the nearest occluding object closer to the object than the light itself?
                    bool inShadow = trace(shadowPointOrig, lightDir, objects, tNearShadow, index, uv, &shadowHitObject) &&
                        tNearShadow * tNearShadow < lightDistance2;
                    lightAmt += (1 - inShadow) * lights[i]->intensity * LdotN;
                    Vec3f reflectionDirection = reflect(-lightDir, N);
                    specularColor += powf(std::max(0.f, -dotProduct(reflectionDirection, dir)), hitObject->specularExponent) * lights[i]->intensity;
                }
                hitColor = lightAmt * hitObject->evalDiffuseColor(st) * hitObject->Kd + specularColor * hitObject->Ks;
                break;
            }
        }
    }

    return hitColor;
}

// [comment]
// The main render function. This where we iterate over all pixels in the image, generate
// primary rays and cast these rays into the scene. The content of the framebuffer is
// saved to a file.
// [/comment]
void render(
    const Options &options,
    const std::vector<std::unique_ptr<Object>> &objects,
    const std::vector<std::unique_ptr<Light>> &lights)
{
    Vec3f *framebuffer = new Vec3f[options.width * options.height];
    Vec3f *pix = framebuffer;
    float scale = tan(deg2rad(options.fov * 0.5));
    float imageAspectRatio = options.width / (float)options.height;
    Vec3f orig(0);
    for (uint32_t j = 0; j < options.height; ++j) {
        for (uint32_t i = 0; i < options.width; ++i) {
            Vec3f colorbuffer(0); // the color buffer to deal with the aliasing problem
            int sampleSize = 50;
            for (int n = 0; n < sampleSize; n++){
                // generate primary ray direction
                float xOffset = ((double) rand() / (RAND_MAX)), yOffset = ((double) rand() / (RAND_MAX));
                float x = (2 * (i + xOffset) / (float)options.width - 1) * imageAspectRatio * scale;
                float y = (1 - 2 * (j + yOffset) / (float)options.height) * scale;
                Vec3f dir = normalize(Vec3f(x, y, -1));
                colorbuffer += castRay(orig, dir, objects, lights, options, 0);
            }
            *(pix++) = colorbuffer / sampleSize;
        }
    }

    // save framebuffer to file
    std::ofstream ofs;
    ofs.open("./out.ppm");
    ofs << "P6\n" << options.width << " " << options.height << "\n255\n";
    for (uint32_t i = 0; i < options.height * options.width; ++i) {
        char r = (char)(255 * clamp(0, 1, framebuffer[i].x));
        char g = (char)(255 * clamp(0, 1, framebuffer[i].y));
        char b = (char)(255 * clamp(0, 1, framebuffer[i].z));
        ofs << r << g << b;
    }

    ofs.close();

    delete [] framebuffer;
}

// [comment]
// In the main function of the program, we create the scene (create objects and lights)
// as well as set the options for the render (image widht and height, maximum recursion
// depth, field-of-view, etc.). We then call the render function().
// [/comment]
int main(int argc, char **argv)
{
    // creating the scene (adding objects and lights)
    std::vector<std::unique_ptr<Object>> objects;
    std::vector<std::unique_ptr<Light>> lights;

    // Spheres
    Sphere *sph1 = new Sphere(Vec3f(0, 0, -12), 2);
    sph1->materialType = DIFFUSE_AND_GLOSSY;
    sph1->diffuseColor = Vec3f(0.9, 0.5, 0.6);
    objects.push_back(std::unique_ptr<Sphere>(sph1));

    Sphere *sph2 = new Sphere(Vec3f(0.5, -2, -7), 0.5);
    sph2->ior = 1.5;
    sph2->materialType = REFLECTION_AND_REFRACTION;
    objects.push_back(std::unique_ptr<Sphere>(sph2));

    // Plane
    Vec3f verts[4] = {{-5,-3,-6}, {5,-3,-6}, {5,-3,-16}, {-5,-3,-16}};
    uint32_t vertIndex[6] = {0, 1, 3, 1, 2, 3};
    Vec2f st[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    MeshTriangle *mesh = new MeshTriangle(verts, vertIndex, 2, st);
    mesh->materialType = DIFFUSE_AND_GLOSSY;
    objects.push_back(std::unique_ptr<MeshTriangle>(mesh));

    // Box
    Vec3f xAxis = Vec3f(0.36, -0.8, 0.48);
    Vec3f yAxis = Vec3f(0.48, 0.60, 0.64);
    Vec3f zAxis = crossProduct(xAxis, yAxis);
    Vec3f centerPts = Vec3f(3, -2, -7);
    float length = 1.5;
    Box* box1 = new Box(centerPts, length, xAxis, yAxis, zAxis);
    box1->materialType = DIFFUSE_AND_GLOSSY;
    box1->diffuseColor = Vec3f(0.1, 0.5, 0.9);
    objects.push_back(std::unique_ptr<Box>(box1));

    // Cylinder
    Vec3f c1 = Vec3f(-2, -3, -7);
    Vec3f c2 = Vec3f(-2, 2, -7);
    float r = 0.5;
    Cylinder* cylinder1 = new Cylinder(c1, c2, r);
    cylinder1->materialType = DIFFUSE_AND_GLOSSY;
    cylinder1->diffuseColor = Vec3f(0.5, 0.9, 0.7);
    objects.push_back(std::unique_ptr<Cylinder>(cylinder1));

    // Lights
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(-20, 70, 20), 0.5))); 
    lights.push_back(std::unique_ptr<Light>(new Light(Vec3f(30, 50, -12), 1))); 
    
    // setting up options
    Options options;
    options.width = 640*2;
    options.height = 480*2;
    options.fov = 70;
    options.maxDepth = 10;
    options.backgroundColor = Vec3f(0.235294, 0.67451, 0.843137);
    options.bias = 0.001;
    
    // finally, render
    render(options, objects, lights);

    return 0;
}
