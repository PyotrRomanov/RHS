__constant float lensSize = 0.04f;
__constant float PI = 3.14159265359f;
 
typedef struct
{
    float3 diffuse;
    float refl;
    float refr;
    bool emissive;
} material;
 
typedef struct
{
    float3 origin;
    float3 direction;
    int objIdx;
    float t;
    float3 N;
    bool inside;
} ray;
 
int map(int x, int in_min, int in_max, int out_min, int out_max)
{
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
 
float3 sampleSkydome(__global float* skydome, float3 D)
{
    float INVPI = 1.0f / PI;
    int u = (int)(2500.0f * 0.5f * (1.0f + atan2( D.x, -D.z ) * INVPI));
    int v = (int)(1250.0f * acos( D.y ) * INVPI);
    int idx = u + v * 2500;
    return (float3)(skydome[idx * 3 + 0], skydome[idx * 3 + 1], skydome[idx * 3 + 2] );
    //return (float3)(1.0f, 1.0f, 1.0f);
}
 
float getRandomFloat(__global float* randoms, int offset)
{
    return randoms[(get_global_id(0) + offset) % 1000];
}
 
ray generateRay(int x, int y, __global float4* camera, uint width, uint height, __global float* randoms)
{
    ray ret;
 
    // Unpacking camera info;
    float3 pos = (float3)(camera[0].x,camera[0].y,camera[0].z);
    float3 p1 = (float3)(camera[1].x,camera[1].y,camera[1].z);
    float3 p2 = (float3)(camera[2].x,camera[2].y,camera[2].z);
    float3 p3 = (float3)(camera[3].x,camera[3].y,camera[3].z);
    float3 up = (float3)(camera[4].x,camera[4].y,camera[4].z);
    float3 right = (float3)(camera[5].x,camera[5].y,camera[5].z);
 
    float r0 = getRandomFloat(randoms, 0);
    float r1 = getRandomFloat(randoms, 1);
    float r2 = getRandomFloat(randoms, 2) - 0.5f;
    float r3 = getRandomFloat(randoms, 3) - 0.5f;
 
    // calculate sub-pixel ray target position on screen plane
    float u = ((float)x + r0) / (float)width;
    float v = ((float)y + r1) / (float)height;
    float3 T = p1 + u * (p2 - p1) + v * (p3 - p1);
    // calculate position on aperture
    float3 P = pos + lensSize * (right + up);
    //float3 P = pos + lensSize * (r2 * right + r3 * up);
    // calculate ray direction
    float3 D = normalize(T - P);
 
    // return new primary ray
    ret.origin = P;
    ret.direction = D;
    ret.objIdx = -1;
    return ret;
}
 
ray intersectSphere(int idx, ray r, float4 s)
{
    float3 L = s.xyz - r.origin;
    float tca = dot(L, r.direction);
    if (tca < 0) return r;
    float d2 = dot(L, L) - tca * tca;
    if (d2 > s.w) return r;
    float thc = sqrt(s.w - d2);
    float t0 = tca - thc;
    float t1 = tca + thc;
    if (t0 > 0)
    {
        if (t0 > r.t) return r;
        r.N = normalize(r.origin + t0 * r.direction - s.xyz);
        r.objIdx = idx;
        r.t = t0;
        return r;
    }
    else
    {
        if ((t1 > r.t)) return r;
        r.N = normalize(s.xyz - (r.origin + t1 * r.direction));
        r.objIdx = idx;
        r.t = t1;
        return r;
    }
}
 
ray intersect(ray r, __global float4* world, uint worldSize)
{
    for (int i = 0; i < worldSize; i++)
        r = intersectSphere(i, r, world[i]);
    return r;
}
 
material GetMaterial(int objIdx, float3 I)
{
    material mat;
    if (objIdx == 0)
    {
        // procedural checkerboard pattern for floor plane
        mat.refl = 0;
        mat.refr = 0;
        mat.emissive = false;
        int tx = ((int)(I.x * 3.0f + 1000) + (int)(I.z * 3.0f + 1000)) & 1;
        mat.diffuse = (float3)(1.0f,1.0f,1.0f) * ((tx == 1) ? 1.0f : 0.2f);
    }
    if ((objIdx == 1) || (objIdx > 8)) { mat.refl = mat.refr = 0; mat.emissive = false; mat.diffuse = (float3)(1.0f,1.0f,1.0f); }
    if (objIdx == 2) { mat.refl = 0.8f; mat.refr = 0; mat.emissive = false; mat.diffuse = (float3)(1, 0.2f, 0.2f); }
    if (objIdx == 3) { mat.refl = 0; mat.refr = 1; mat.emissive = false; mat.diffuse = (float3)(0.9f, 1.0f, 0.9f); }
    if (objIdx == 4) { mat.refl = 0.8f; mat.refr = 0; mat.emissive = false; mat.diffuse = (float3)(0.2f, 0.2f, 1); }
    if ((objIdx > 4) && (objIdx < 8)) { mat.refl = mat.refr = 0; mat.emissive = false; mat.diffuse = (float3)(1.0f,1.0f,1.0f); }
    if (objIdx == 8) { mat.refl = mat.refr = 0; mat.emissive = true; mat.diffuse = (float3)(8.5f, 8.5f, 7.0f); }
    return mat;
}
 
float3 reflect(float3 V, float3 N)
{
    return V - 2.0f * dot(V, N) * N;
}
 
float3 refract(bool inside, float3 D, float3 N, float3 R, float rnd)
{
    float nc = inside ? 1 : 1.2f, nt = inside ? 1.2f : 1;
    float nnt = nt / nc, ddn = dot(D, N);
    float cos2t = 1.0f - nnt * nnt * (1 - ddn * ddn);
    R = reflect(D, N);
    if (cos2t >= 0)
    {
        float r1 = rnd;
        float a = nt - nc;
        float b = nt + nc;
        float R0 = a * a / (b * b);
        float c = 1 + ddn;
        float Tr = 1 - (R0 + (1 - R0) * c * c * c * c * c);
        if (r1 < Tr) R = (D * nnt - N * (ddn * nnt + sqrt(cos2t)));
    }
    return R;
}
 
float3 diffuseRefract(float3 N, __global float* randoms)
{
    float r1 = getRandomFloat(randoms, 6);
    float r2 = getRandomFloat(randoms, 7);
    float r = sqrt(1.0f - r1 * r1);
    float phi = 2 * PI * r2;
    float3 R;
    R.x = cos(phi) * r;
    R.y = sin(phi) * r;
    R.z = r1;
    if (dot(N, R) < 0) R *= -1.0f;
    return R;
}
 
// sample: samples a single path up to a maximum depth
float3 sample(ray r, int depth, __global float4* world, __global float* skydome, uint worldSize, __global float* randoms)
{
    r = intersect(r, world, worldSize);
 
    if (r.objIdx == -1)
        return sampleSkydome(skydome, r.direction);
 
    float3 I = r.origin + r.t * r.direction;
    material mat = GetMaterial(r.objIdx, I);
 
    if (mat.emissive) return mat.diffuse;
    if (depth >= 20) return (float3)(0.0f, 0.0f, 0.0f);
 
    float r0 = getRandomFloat(randoms, 0);
    float3 R = (float3)(0.0f, 0.0f, 0.0f);
    if (r0 < mat.refr)
    {
        R = refract(r.inside, r.direction, r.N, R, getRandomFloat(randoms, 5));
        ray er;
        er.origin = I + R * 0.0001f;
        er.direction = R;
        er.inside = (dot(r.N, R) < 0);
        return mat.diffuse; // Recursive
    }
    else if ((r0 < (mat.refl + mat.refr)) && (depth < 20))
    {
        // pure specular reflection
        R = reflect(r.direction, r.N);
        ray er;
        er.origin = I + R * 0.0001f;
        er.direction = R;
        return mat.diffuse;// Recursiev shit * Sample(extensionRay, depth + 1);
    }
    else
    {
        // diffuse reflection
        R = diffuseRefract(r.N, randoms);
        ray er;
        er.origin = I + R * 0.0001f;
        er.direction = R;
        return dot(R, r.N) * mat.diffuse;// Recursive shit * Sample(extensionRay, depth + 1);
    }
}
 
__kernel void Main(__global int* dst, uint width, uint height, __global float4* world, uint worldSize, __global float* skydome, __global float4* camera, __global float* randoms)
{
    // dst = output buffer
    // width & height = dimensions of the output buffer
    // world = pos + radius;
 
    int idx = get_global_id(0);// * get_global_size(0);
    int idy = get_global_id(1);// * get_global_size(1);
 
 
    ray r = generateRay(idx, idy, camera, width, height, randoms);
    float3 col = sample(r, 0, world, skydome, worldSize, randoms);
 
    dst[idx + idy * width] = ((int)(col.x * 255) << 16) + ((int)(col.y * 255) << 8) + (int)(col.z * 255);
}