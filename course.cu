//C, C++ LIBRARIES HEADERS
#include <cstdio>
#include <chrono>

//OPENGL HEADERS
#include <GL/glew.h>
#include <GL/freeglut.h>

//CUDA HEADERS
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

//SOME MACROS AND DEFINES
#define INF_DOUBLE 1.e8
#define GRAVITY_EPS 1.e-3

#define sqr(x) ((x)*(x))
#define CSC(call) {                            \
    cudaError err = call;                        \
    if(err != cudaSuccess) {                        \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                            \
    }                                    \
} while (0)

//functions definitions
__host__ __device__ double targ_f(double2 point);

//GLOBAL PARAMS
int n_blocks = 32;
int n_threads = 32;

__device__ double d_dx; //workaround to read from functor



//TYPES AND FUNCTORS

typedef struct {
    double2 x,          //current coordinate
            v,          //current speed
            best_x,     //current best local minimum
            f;          //current force applied
} particle_t;

typedef struct {
    __device__ bool operator()(particle_t left, particle_t right){
        int l_bl_x,
            l_bl_y,
            r_bl_x,
            r_bl_y;

        l_bl_x = floor(left.x.x / d_dx);
        l_bl_y = floor(left.x.y / d_dx);
        r_bl_x = floor(right.x.x / d_dx);
        r_bl_y = floor(right.x.y / d_dx);

        return (l_bl_x == r_bl_x) ? l_bl_y < r_bl_y : l_bl_x < r_bl_x;
    }
} particle_comparator_bl;

typedef struct {
    __host__ __device__ bool operator()(particle_t left, particle_t right){
        return targ_f(left.best_x) < targ_f(right.best_x);
    }
} particle_comparator_function;

typedef struct {
    __host__ __device__ particle_t operator()(particle_t left, particle_t right) {
        particle_t res;
        res.x.x = left.x.x + right.x.x;
        return res;
    }
} summator_x;

typedef struct {
    __host__ __device__ particle_t operator()(particle_t left, particle_t right) {
        particle_t res;
        res.x.y = left.x.y + right.x.y;
        return res;
    }
} summator_y;

//GLOBAL VARIABLES
//because of openGL API
//i wonder who thought of "void (*func)(void)" signature
summator_x sum_x;                           //summator for reducer
summator_y sum_y;                           //summator for reducer
int width,                                  //openGL params
    height,
    n,                                      //SPO params
    grid_on = 1,                            //turn on/off grid
    time_on = 1;                            //turn on/off time
double *fa,                                 //function's value array
       scale_x,                             //axis scale
       scale_y,
       w,                                   //SPO params 
       a1, 
       a2, 
       dt, 
       dx, 
       g;
__device__ double grid_zero_dist = 0.05;    //magic number to draw grid
particle_t *pa;                             //particles array
curandState *rsa;                           //random states array
thrust::device_ptr<particle_t> th_pa;       //thrust's pointer to particles array
thrust::device_ptr<double> th_fa;           //thrust's pointer to function's value array
double2 gvbest_x;                           //global best coordinates
GLuint vbo;                                 //opengl stuff
struct cudaGraphicsResource *cuda_res;      //cuda + opengl interop stuff


/////////////////////////////////////////////////
/////////DEVICE FUNCTIONS START//////////////////
/////////////////////////////////////////////////

__host__ __device__ double targ_f(double2 point){
    return -20*exp(-.2*sqrt(.5*(sqr(point.x) + sqr(point.y)))) - 
           exp(.5*(cospi(2*point.x) + cospi(2*point.y))) + exp(1.) + 20;
}

/////////////////////////////////////////////////
/////////DEVICE FUNCTIONS END////////////////////
/////////////////////////////////////////////////
/////////KERNEL FUNCTIONS START//////////////////
/////////////////////////////////////////////////

__global__ void k_calc_abs_heatmap(double *fa,
                                   int w,
                                   int h,
                                   double scale_x,
                                   double scale_y,
                                   double2 center){
    int idx, ofs, i, pxlno, row, col;
    double f;
    double2 point;
    
    idx = threadIdx.x + blockIdx.x * blockDim.x;
    ofs = blockDim.x * gridDim.x;
    pxlno = w * h;

    for (i = idx; i < pxlno; i += ofs){
        row = i / w;
        col = i % w;

        point = double2();
        point.x = (2*col / (double)(w-1) -1.) * scale_x + center.x;
        point.y = -(2*row / (double)(h-1) -1.) * scale_y + center.y;
        f = targ_f(point);

        fa[i] = f;
    }
}

__global__ void k_draw_norm_heatmap(uchar4 *pxla,
                                    double *fa,
                                    int w,
                                    int h,
                                    double min_f,
                                    double max_f){
    int idx, ofs, i, pxlno;
    double f;
    
    idx = threadIdx.x + blockIdx.x * blockDim.x;
    ofs = blockDim.x * gridDim.x;
    pxlno = w * h;
    max_f -= min_f;

    for (i = idx; i < pxlno; i += ofs){
        f = fa[i];
        f = (f - min_f) / max_f;

        pxla[i] = make_uchar4(f*255, 0, 0, 255);
    }
}

__global__ void k_draw_particles(uchar4 *pxla,
                                 int w,
                                 int h,
                                 double scale_x,
                                 double scale_y,
                                 double2 center,
                                 particle_t *pa,
                                 int n){
    int idx, ofs, i;
    double2 point;
    int2 int_point;
    
    idx = threadIdx.x + blockIdx.x * blockDim.x;
    ofs = blockDim.x * gridDim.x;
    point = double2();
    int_point = int2();

    for (i = idx; i < n; i += ofs){
        point = pa[i].x;

        point.x = ((point.x - center.x) / scale_x + 1) * (w - 1) / 2;
        point.y = (-1*(point.y - center.y) / scale_y + 1) * (h - 1) / 2;
        int_point.x = (int)point.x;
        int_point.y = (int)point.y;

        if (int_point.x > 0 && int_point.x < w && int_point.y > 0 && int_point.y < h){
            pxla[int_point.y * w + int_point.x] = make_uchar4(255, 255, 255, 255);
        }
    }
}

__global__ void k_draw_grid(uchar4 *pxla,
                            int w,
                            int h,
                            double scale_x,
                            double scale_y,
                            double2 center,
                            double dx){
    int idx, ofs, i, pxlno, row, col, bnox, bnoy;
    double2 point, lb, rb;
    
    idx = threadIdx.x + blockIdx.x * blockDim.x;
    ofs = blockDim.x * gridDim.x;
    pxlno = w*h;

    for (i = idx; i < pxlno; i += ofs){
        row = i / w;
        col = i % w;
        
        point = double2();
        point.x = (2*col / (double)(w-1) -1.) * scale_x + center.x;
        point.y = -(2*row / (double)(h-1) -1.) * scale_x + center.y;

        bnox = point.x / dx;
        bnoy = point.y / dx;
        if (point.x < 0) --bnox;
        if (point.y < 0) --bnoy;
        lb.x = dx*(bnox);
        lb.y = dx*(bnoy);
        rb.x = dx*(bnox+1);
        rb.y = dx*(bnoy+1);
        if (abs(point.x - lb.x) < grid_zero_dist ||
            abs(point.y - lb.y) < grid_zero_dist ||
            abs(point.x - rb.x) < grid_zero_dist ||
            abs(point.y - rb.y) < grid_zero_dist){

            pxla[i] = make_uchar4(0, 0, 255, 64);
        }
        
    }
}

__global__ void k_init(particle_t *pa, 
                       curandState *rsa,
                       int pn, 
                       double dx){
    int idx, ofs, i;
    particle_t loc_p;

    idx = threadIdx.x + blockIdx.x * blockDim.x;
    ofs = blockDim.x * gridDim.x;

    for (i = idx; i < pn; i += ofs){
        loc_p = particle_t();
        curand_init(1337, 
                    i,
                    0,
                    rsa+i);

        loc_p.x = double2();
        loc_p.x.x = 10 + curand_uniform(rsa+i) * 5;
        loc_p.x.y = 10 + curand_uniform(rsa+i) * 5;
        loc_p.v = double2();
        loc_p.best_x.x = loc_p.x.x;
        loc_p.best_x.y = loc_p.x.y;
        loc_p.f = make_double2(0., 0.);

        pa[i] = loc_p;
    }
}

__global__ void k_calc_forces(particle_t *pa, 
                              thrust::device_ptr<particle_t> th_pa, 
                              int n, 
                              double dx){
    extern __shared__ size_t bords[];       // allocated on invocation
    int idx, ofs, i, j, lb, rb;
    double dist_x, dist_y, dist_pow4;
    particle_t loc_p, loc_adjp;
    particle_comparator_bl comp_bl;

    idx = threadIdx.x + blockIdx.x * blockDim.x;
    ofs = blockDim.x * gridDim.x;

    for (i = idx; i < n; i += ofs){
        loc_p = pa[i];
        loc_p.f.x = 0.;
        loc_p.f.y = 0.;

        //get borders of particle's block
        //fs' summation operator in space, generated by blocks' division, is considered continious by right
        loc_adjp = loc_p;
        loc_adjp.x.x -= d_dx;
        loc_adjp.x.y -= d_dx;
        lb = (int)(thrust::lower_bound(thrust::device,
                                 th_pa, 
                                 th_pa + n,
                                 loc_adjp,
                                 comp_bl) - th_pa);
        loc_adjp.x.x += d_dx;
        loc_adjp.x.y += d_dx;
        rb = (int)(thrust::upper_bound(thrust::device,
                                 th_pa, 
                                 th_pa + n,
                                 loc_adjp,
                                 comp_bl) - th_pa);
        for (j = lb; j < rb; ++j){
            loc_adjp = pa[j];
            dist_x = loc_adjp.x.x - loc_p.x.x;
            dist_y = loc_adjp.x.y - loc_p.x.y;
            dist_pow4 = sqr(sqr(dist_x) + sqr(dist_y));
            loc_p.f.x += dist_x / (dist_pow4 + GRAVITY_EPS);
            loc_p.f.y += dist_y / (dist_pow4 + GRAVITY_EPS);
        }

        //calculate left semirange of mid column
        for (j = i-1; j >= 0; --j){
            loc_adjp = pa[j];

            //check if we got out of range in OX or OY axis
            if (floor(loc_adjp.x.x / dx) < floor(loc_p.x.x / dx) ||
                floor(loc_adjp.x.y / dx) < floor(loc_p.x.y / dx)-1) break;

            //add the f
            dist_x = loc_adjp.x.x - loc_p.x.x;
            dist_y = loc_adjp.x.y - loc_p.x.y;
            dist_pow4 = sqr(sqr(dist_x) + sqr(dist_y));
            loc_p.f.x += dist_x / (dist_pow4 + GRAVITY_EPS);
            loc_p.f.y += dist_y / (dist_pow4 + GRAVITY_EPS);
        }
        //calculate right semirange of mid column
        for (j = i+1; j < n; ++j){
            loc_adjp = pa[j];

            //check if we got out of range in OX or OY axis
            if (floor(loc_adjp.x.x / dx) > floor(loc_p.x.x / dx) ||
                floor(loc_adjp.x.y / dx) > floor(loc_p.x.y / dx)+1) break;

            //add the f
            dist_x = loc_adjp.x.x - loc_p.x.x;
            dist_y = loc_adjp.x.y - loc_p.x.y;
            dist_pow4 = sqr(sqr(dist_x) + sqr(dist_y));
            loc_p.f.x += dist_x / (dist_pow4 + GRAVITY_EPS);
            loc_p.f.y += dist_y / (dist_pow4 + GRAVITY_EPS);
        }

        //count right column
        loc_adjp = loc_p;
        loc_adjp.x.x += d_dx;
        loc_adjp.x.y -= d_dx;
        lb = (int)(thrust::lower_bound(thrust::device,
                                 th_pa, 
                                 th_pa + n,
                                 loc_adjp,
                                 comp_bl) - th_pa);
        loc_adjp.x.x += d_dx;
        loc_adjp.x.y += d_dx;
        rb = (int)(thrust::upper_bound(thrust::device,
                                 th_pa, 
                                 th_pa + n,
                                 loc_adjp,
                                 comp_bl) - th_pa);
        for (j = lb; j < rb; ++j){
            loc_adjp = pa[j];
            dist_x = loc_adjp.x.x - loc_p.x.x;
            dist_y = loc_adjp.x.y - loc_p.x.y;
            dist_pow4 = sqr(sqr(dist_x) + sqr(dist_y));
            loc_p.f.x += dist_x / (dist_pow4 + GRAVITY_EPS);
            loc_p.f.y += dist_y / (dist_pow4 + GRAVITY_EPS);
        }

        pa[i] = loc_p;
    }
}

__global__ void k_iterate(particle_t *pa,
                          curandState *rsa,
                          int n,
                          double w,
                          double a1,
                          double a2,
                          double g,
                          double dt,
                          double2 gbest,
                          double dx){
    int idx, ofs, i;
    double r1, r2, target_func;
    particle_t loc_p;
    curandState loc_state;

    idx = threadIdx.x + blockIdx.x * blockDim.x;
    ofs = blockDim.x * gridDim.x;
    
    for (i = idx; i < n; i += ofs){
        loc_p = pa[i];
        loc_state = rsa[i];
        
        r1 = curand_uniform(&loc_state);
        r2 = curand_uniform(&loc_state);

        //update velocity
        loc_p.v.x = w * loc_p.v.x +
                    dt * (a1 * r1 * (loc_p.best_x.x - loc_p.x.x) + 
                          a2 * r2 * (gbest.x - loc_p.x.x) - 
                          g * loc_p.f.x);
        loc_p.v.y = w * loc_p.v.y +
                    dt * (a1 * r1 * (loc_p.best_x.y - loc_p.x.y) + 
                          a2 * r2 * (gbest.y - loc_p.x.y) - 
                          g * loc_p.f.y);

        //update coordinates
        loc_p.x.x += loc_p.v.x * dt;
        loc_p.x.y += loc_p.v.y * dt;

        //update best coordinates
        target_func = targ_f(loc_p.x);
        if (target_func < targ_f(loc_p.best_x)){
            loc_p.best_x = loc_p.x;
        }

        rsa[i] = loc_state;
        pa[i] = loc_p;
    }
}

/////////////////////////////////////////////////
/////////KERNEL FUNCTIONS END////////////////////
/////////////////////////////////////////////////
/////////HOST FUNCTIONS START////////////////////
/////////////////////////////////////////////////

void update_scene(){
    uchar4 *pxla;
    double2 gbest_x, cam_center;
    particle_t gbest_particle, aaaaa;
    size_t sz;
    double scene_min_f, scene_max_f, gbest;
    particle_comparator_function comp_f;        //comparator for min/max
    particle_comparator_bl comp_bl;             //comparator for sort
    auto s_time = std::chrono::high_resolution_clock::now();
    auto e_time = std::chrono::high_resolution_clock::now();

    //bind array to cuda
    CSC(cudaGraphicsMapResources(1, &cuda_res, 0));
    CSC(cudaGraphicsResourceGetMappedPointer((void**) &pxla, &sz, cuda_res));

    //get masses' center
    aaaaa.x.x = 0;
    aaaaa = thrust::reduce(thrust::device, th_pa, th_pa + n, aaaaa, sum_x);
    cam_center.x = aaaaa.x.x / n;
    aaaaa.x.y = 0;
    aaaaa = thrust::reduce(thrust::device, th_pa, th_pa + n, aaaaa, sum_y);
    cam_center.y = aaaaa.x.y / n;

    //draw scene
    k_calc_abs_heatmap<<< n_blocks, n_threads >>>(fa,
                                                  width,
                                                  height,
                                                  scale_x,
                                                  scale_y,
                                                  cam_center);

    scene_min_f = *thrust::min_element(thrust::device, th_fa, th_fa + width*height);
    scene_max_f = *thrust::max_element(thrust::device, th_fa, th_fa + width*height);

    k_draw_norm_heatmap<<< n_blocks, n_threads >>>(pxla,
                        fa,
                        width,
                        height,
                        scene_min_f,
                        scene_max_f);

    k_draw_particles<<< n_blocks, n_threads >>>(pxla,
                     width,
                     height,
                     scale_x,
                     scale_y,
                     cam_center,
                     pa,
                     n);

    if (grid_on) k_draw_grid<<< n_blocks, n_threads >>>(pxla,
                                                        width,
                                                        height,
                                                        scale_x,
                                                        scale_y,
                                                        cam_center,
                                                        dx);

    //if time is off, leave scene as it is
    if (!time_on){
        glutPostRedisplay();
        //unbind all
        CSC(cudaDeviceSynchronize());
        CSC(cudaGraphicsUnmapResources(1, &cuda_res, 0));
        return;
    }
    
    //sort particles in order to make faster fs' calculation by 
    ////using space's division
    thrust::sort(thrust::device, th_pa, th_pa+n, comp_bl);
    gbest_particle = *thrust::min_element(thrust::device, th_pa, th_pa+n, comp_f);
    gbest = targ_f(gbest_particle.best_x);
    gbest_x = gbest_particle.best_x;
    if (gbest < targ_f(gvbest_x)){
        gvbest_x = gbest_x;
    }
    //next iteration
    k_calc_forces<<< n_blocks, n_threads, sizeof(size_t)*2 >>>(pa, 
                                                               th_pa, 
                                                               n, 
                                                               dx);
    k_iterate<<< n_blocks, n_threads >>>(pa, 
                                         rsa,
                                         n, 
                                         w, 
                                         a1, 
                                         a2, 
                                         g,
                                         dt,
                                         gvbest_x,
                                         dx);

    //unbind all
    CSC(cudaDeviceSynchronize());
    CSC(cudaGraphicsUnmapResources(1, &cuda_res, 0));
    e_time = std::chrono::high_resolution_clock::now();
    printf("Best coords: x=%lf y=%lf\n", gvbest_x.x, gvbest_x.y);
    printf("Done in %ldmcs\n", std::chrono::duration_cast<std::chrono::microseconds>(e_time - s_time).count());

    //call draw
    glutPostRedisplay();
}

void display_func() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);    
    glutSwapBuffers();
}

void keys_handler(unsigned char key, int x, int y){
    switch (key){
    case '-':
        scale_x += 0.5;
        scale_y = scale_x * height / width;
        break;
    case '+':
        if (scale_x > 1){
            scale_x -= 0.5;
            scale_y = scale_x * height / width;
        }
        break;
    case '/':
        grid_on = (grid_on + 1) % 2;
        break;
    case ' ':
        time_on = (time_on + 1) % 2;
        break;
    };
}

void init_all(){
    double h_gzd = 0.05;
    particle_t gbest_particle;
    particle_comparator_function comp_f;        //comparator for min/max

    CSC(cudaMemcpyFromSymbol(&h_gzd, grid_zero_dist, sizeof(grid_zero_dist), 0, cudaMemcpyDeviceToHost));

    k_init<<< n_blocks, n_threads >>>(pa, 
                                      rsa, 
                                      n, 
                                      dx);
    gbest_particle = *thrust::min_element(thrust::device, th_pa, th_pa+n, comp_f);
    gvbest_x = gbest_particle.best_x;
    h_gzd *= dx;
    CSC(cudaMemcpyToSymbol(grid_zero_dist, &h_gzd, sizeof(h_gzd), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(d_dx, &dx, sizeof(dx), 0, cudaMemcpyHostToDevice));
}

/////////////////////////////////////////////////
/////////MAIN////////////////////////////////////
/////////////////////////////////////////////////

int main(int argc, char **argv){
    //declare variables to use

    //read, init variables
    width = 800;
    height = 600;
    scale_x = 5.;
    scale_y = scale_x * height / width;
    printf("Number of particles (n) = ");
    scanf("%d", &n);
    printf("Inertia (w) = ");
    scanf("%lf", &w);
    printf("a1 = ");
    scanf("%lf", &a1);
    printf("a2 = ");
    scanf("%lf", &a2);
    printf("dx = ");
    scanf("%lf", &dx);
    printf("Time delta (dt) = ");
    scanf("%lf", &dt);
    printf("Antigravity f coefficient (g) = ");
    scanf("%lf", &g);

    //allocate enough memory
    CSC(cudaMalloc(&pa, sizeof(*pa) * n));
    CSC(cudaMalloc(&fa, sizeof(*fa) * width * height));
    CSC(cudaMalloc(&rsa, sizeof(*rsa) * n));
    th_pa = thrust::device_pointer_cast(pa);
    th_fa = thrust::device_pointer_cast(fa);

    //init particles and grid size
    init_all();

    //init opengl stuff
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("Heat map");
    glutIdleFunc(update_scene);
    glutDisplayFunc(display_func);
    glutKeyboardFunc(keys_handler);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);

    glewInit();
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

    //bind CUDA array pointer to openGL array pointer
    CSC(cudaGraphicsGLRegisterBuffer(&cuda_res, vbo, cudaGraphicsMapFlagsWriteDiscard));

    //start openGL main loop
    glutMainLoop();

    //dispose used resources
    CSC(cudaGraphicsUnregisterResource(cuda_res));
    glBindBuffer(1, vbo);
    glDeleteBuffers(1, &vbo);

    //free all memory left
    CSC(cudaFree(pa));
    CSC(cudaFree(fa));
    CSC(cudaFree(rsa));

    return 0;
}
