KERNEL(warm_up_gpu)(int c, int a, int b, __global int* out)
{
    int res = (get_global_id(0) * a + get_global_id(1)) * b + get_global_id(2);
    if(a >> 3)
        res += get_local_id(1);
    if(c)
        out[get_local_id(0)] = res;
}