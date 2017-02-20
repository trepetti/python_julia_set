// Modulus of the complex value.
inline float cmod(float2 a){
    return (sqrt(a.x * a.x + a.y * a.y));
}

// Complex multiplication.
inline float2 cmul(float2 a, float2 b)
{
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__kernel void julia(__global const float2 *domain_gpu, const float2 c, unsigned char max_iterations,
                    __global unsigned char *codomain_gpu)
{
    int i, j;
    unsigned char n;
    float2 z;

    // Find the index of this work item.
    i = get_global_id(0);

    // Look up the corresponding element of the domain.
    z = domain_gpu[i];

    // Identical to Python version.
    n = 0;
    while (n < max_iterations && cmod(z) < 2.0) {
        z = cmul(z, z) + c;
        n++;
    }

    // Again, should be a value between 0 and max_iterations.
    codomain_gpu[i] = n;
}