// XXX XXX XXX CAREFUL WITH IMAGINARY TIME EVOLUTION: MUST BE tau=-it

//#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <time.h>
//#include <cuda.h> // XXX REALLY NECESSARY?

#define SHARED_SIZE_PRIMARY 28
#define SHARED_SIZE_SECONDARY 14

void evolve2d(float *ket, float t, float deltat, float *energias, int order, int Nx, int Ny, float Vx, float Vy, int pbcx, int pbcy, float sinkxax, float coskxax, float sinkyay, float coskyay); 
void evolve2dO2(float *ket, float t, float deltat, float *energias, int Nx, int Ny, float Vx, float Vy, int pbcx, int pbcy, float kxax, float coskxax, float sinkyay, float kyay);
void H2d(float *ket, float deltat, int id, int Nx, int Ny, float Vx, float Vy, int pbcx, int pbcy, float sinkxax, float coskxax, float sinkyay, float coskyay);
__global__ void H2d_step(float *ket_in, float *ket_out, int dataketid, float deltat, float *prepared_energias, int pbcx, int pbcy, int Nx, int Ny, float Vx, float Vy, float sinkxax, float coskxax, float sinkyay, float coskyay);
__global__ void H2d_x(float *ket, float deltat, int id, int pbcx, int Nx, int Ny, float Vx, float sinkxax, float coskxax);
__global__ void H2d_y(float *ket, float deltat, int id, int pbcy, int Nx, int Ny, float Vy, float sinkyay, float coskyay);
__global__ void Hdiag2d(float *ket, float deltat, float *energias, int Nx, int Ny, float Vx, float Vy);
//__global__ void H2di_x(float *ket, float deltatau, int id, int pbcx, int Nx, int Ny, float Vx, float sinkxax, float coskxax);
//__global__ void H2di_y(float *ket, float deltatau, int id, int pbcy, int Nx, int Ny, float Vy, float sinkyay, float coskyay);
//__global__ void Hdiag2di(float *ket, float deltatau, float *energias, int Nx, int Ny, float Vx, float Vy);

// Definition of variables {{{
/*
	int		debug;
	float	pi;
	float hbar;
	float	dNx;
	float	dNy;
	float	Lx;
	float	Ly;
	//int 	pbcx;
	//int 	pbcy;
	float	lambda;	
	float	innerR;
	float	outerR;
	float	alpha;
	float	beta;
	float	dt;
	float	dtmax;
	int		order;
	int		gsteps;
	int		verbose;
	int		debut;
	int		ground;
	int		nodyn;
	int		pars;
	char*	filepars;
	float	cloakEnergy;
	float	tmax;
	float	mass0;
	//float	ax;
	//float	ay;
	//float	kx;
	//float	ky;
*/
//}}}

// Definition of functions { { {
void evolve2d(float *ket, float t, float deltat, float *energias, int order, int Nx, int Ny, float Vx, float Vy, int pbcx, int pbcy, float sinkxax, float coskxax, float sinkyay, float coskyay) {{{
{
	if (order==2) evolve2dO2(ket, t, deltat, energias, Nx, Ny, Vx, Vy, pbcx, pbcy, sinkxax, coskxax, sinkyay, coskyay);
//	if (order==4) evolve2dO4(ket, t, deltat, energias, Nx, Ny, Vx, Vy, sinkxax, coskxax, sinkyay, coskyay);
} }}}
void evolve2dO2(float *ket, float t, float deltat, float *energias, int Nx, int Ny, float Vx, float Vy, int pbcx, int pbcy, float sinkxax, float coskxax, float sinkyay, float coskyay) {{{
//	This function leads the real-time evolution in second order up to time t.
//	float	ket			1D array describing the ket always alternating real and imaginary part of the ket
//	float	t			duration of the evolution
//	float	deltat		incremental timestep
//	float	energias*	Potential surface (e.g. external trapping potential) acting on the condensate
//	int		Nx			Number of discretization points in x direction
//	int		Ny			Number of discretization points in y direction
//	float	Vx			Off-disgonal matrix terms in x direction
//	float	Vy			Off-disgonal matrix terms in y direction
//	float	sinkxax		sin(kx*ax) - for better performance
//	float	coskxax		cos(kx*ax) - for better performance
//	float	sinkyay		sin(kx*ax) - for better performance
//	float	coskyay		cos(kx*ax) - for better performance
{
	// Threadsize depends on local (cache) memory.
	// My cards have only 16kB per block.
	// Each floating point takes 4 bits x 2 for complex numbers
	// Therefore, 32*32*8 = 8kB is already quite a lot.
	// XXX TRY LARGER VALUES
	// XXX 45x45*8 = 16200 < 16384
	int threadsPerBlockX         = SHARED_SIZE_PRIMARY;
	int threadsPerBlockY         = SHARED_SIZE_SECONDARY;
	int overhead                 = 2;
	int effectiveThreadsPerBlock = threadsPerBlockX - 2*overhead;
	int blocksPerGridX = (Nx + effectiveThreadsPerBlock - 1) / effectiveThreadsPerBlock;
	int blocksPerGridY = (Ny + effectiveThreadsPerBlock - 1) / effectiveThreadsPerBlock;
	dim3 threadsPerBlock(threadsPerBlockX, threadsPerBlockY);
	dim3 blocksPerGrid(blocksPerGridX, blocksPerGridY);
	
	int Nbrofits = (int) (t/deltat);
	float Remainingtime=t-((float)Nbrofits)*deltat;
	
	//int *inputarray = (int*) malloc(blocksPerGridX*blocksPerGridY*sizeof(int));
	//for (int i=0; i<blocksPerGridX*blocksPerGridY; i++) inputarray[i] = 0;

	float *prepared_energias = (float*) malloc(2*Nx*Ny*sizeof(float));
	for (int i=0; i<Nx*Ny; i++)
	{
		prepared_energias[2*i]   = cos(-deltat*(energias[i]-2.0*Vx-2.0*Vy));
		prepared_energias[2*i+1] = sin(-deltat*(energias[i]-2.0*Vx-2.0*Vy));
		//prepared_energias[2*i]   = 1.0;
		//prepared_energias[2*i+1] = 0.0;
	}

	float *d_ket0;
	float *d_ket1;
	float *d_prepared_energias;
	//int *d_inputarray;
	cudaMalloc((void**)&d_ket0, 2*Nx*Ny*sizeof(float));
	cudaMalloc((void**)&d_ket1, 2*Nx*Ny*sizeof(float));
	cudaMalloc((void**)&d_prepared_energias, 2*Nx*Ny*sizeof(float));
	//cudaMalloc((void**)&d_inputarray, blocksPerGridX*blocksPerGridY*sizeof(int));
    cudaMemcpy(d_ket0, ket, 2*Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ket1, ket, 2*Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prepared_energias, prepared_energias, 2*Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_inputarray, inputarray, blocksPerGridX*blocksPerGridY*sizeof(int), cudaMemcpyHostToDevice);
	
	
	bool DoRemainder;
	if (Remainingtime/deltat > 0.001) DoRemainder = true;
	else DoRemainder = false;
	float dt;
	float dthalf;
	
	dt     = deltat;
	dthalf = deltat/2.0;
	int dataketid=0;
	//printf("%f %d %d %d %d %f %f %f %f %f %f %f %f\n", dt, pbcx, pbcy, Nx, Ny, Vx, Vy, sinkxax, coskxax, sinkyay, coskyay);
	for (int i=0; i<Nbrofits; i++)
	//for (int i=0; i<1; i++)
	{
		H2d_step<<<blocksPerGrid, threadsPerBlock>>>(d_ket0, d_ket1, dataketid, dt, d_prepared_energias, pbcx, pbcy, Nx, Ny, Vx, Vy, sinkxax, coskxax, sinkyay, coskyay);
		cudaThreadSynchronize();
		dataketid=1-dataketid;
	}
	
//	if (DoRemainder)
//	{
//		dt     = Remainingtime;
//		H2d_step<<<blocksPerGrid, threadsPerBlock>>>(d_ket, dt, energias, pbcx, pbcy, Nx, Ny, Vx, Vy, sinkxax, coskxax, sinkyay, coskyay);		// x-axis, odd
//	}
	
	if (dataketid == 0) cudaMemcpy(ket, d_ket0, 2*Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
	else cudaMemcpy(ket, d_ket1, 2*Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_ket0);
	cudaFree(d_ket1);
	cudaFree(d_prepared_energias);
} }}}
__global__ void H2d_step(float *ket0, float *ket1, int dataketid, float deltat, float* prepared_energias, int pbcx, int pbcy, int Nx, int Ny, float Vx, float Vy, float sinkxax, float coskxax, float sinkyay, float coskyay) {{{
{
	// Two ket positions on each side need to be calculated for
	// intermediate steps but cannot be returned as result
	int overhead  = 2;
	// The number of threads to be returned from this block
	// is the number of input threads minus the overhead on each side
	// Note taht the number of ket elements will be (twice=complex) the
	// square of this number
	//int effectiveSize = SHARED_SIZE_PRIMARY - 2*overhead;
	// The local storage block has the same size as the 
	__shared__ float shared_ket[2*SHARED_SIZE_PRIMARY*SHARED_SIZE_PRIMARY];
	
	int globalx0 = (SHARED_SIZE_PRIMARY-2*overhead)*blockIdx.x - overhead;
	int globaly0 = (SHARED_SIZE_PRIMARY-2*overhead)*blockIdx.y - overhead;
	
	int globalid = ((globalx0 + threadIdx.x))*Ny + ((globaly0+2*threadIdx.y));
	int globalidpbc0 = ((Nx + globalx0 + threadIdx.x)%Nx)*Ny + ((Ny+globaly0+2*threadIdx.y)%Ny);
	int globalidpbc1 = ((Nx + globalx0 + threadIdx.x)%Nx)*Ny + ((Ny+globaly0+2*threadIdx.y+1)%Ny);
	int sharedid = threadIdx.x*SHARED_SIZE_PRIMARY + 2*threadIdx.y;
	int spos = 2*(sharedid);
	int gpos = 2*(globalid);
	int gpospbc0 = 2*(globalidpbc0);
	int gpospbc1 = 2*(globalidpbc1);
	if ( dataketid == 0 )
	{
		shared_ket[spos]   = ket0[gpospbc0];
		shared_ket[spos+1] = ket0[gpospbc0+1];
		shared_ket[spos+2] = ket0[gpospbc1];
		shared_ket[spos+3] = ket0[gpospbc1+1];
	} else {
		shared_ket[spos]   = ket1[gpospbc0];
		shared_ket[spos+1] = ket1[gpospbc0+1];
		shared_ket[spos+2] = ket1[gpospbc1];
		shared_ket[spos+3] = ket1[gpospbc1+1];
	}	
//	if (!(pbcx)) // XXX NOT IMPLEMENTED YET
//	if (!(pbcy)) // XXX NOT IMPLEMENTED YET
//	{	
//		shared_ket[spos]   = 0.0;
//		shared_ket[spos+1] = 0.0;
//		shared_ket[spos+2] = 0.0;
//		shared_ket[spos+3] = 0.0;
//	}
	__syncthreads();
					
	int mx, my;
	
	float temp1re, temp1im, temp2re, temp2im;
	float ketre, ketim;
	int cap, cap_p;
	
	float ketre_p, ketim_p;
	
	float deltathalf = deltat/2.0;
	
	float costx=(float) cos(deltathalf*Vx);
	float sintx=(float) sin(deltathalf*Vx);
	
	float costy=(float) cos(deltathalf*Vx);
	float sinty=(float) sin(deltathalf*Vy);
	
	int rap;
	float kre, kim, tmp, costmp, sintmp;
	
	my = threadIdx.x;
	mx = 2*threadIdx.y;
	cap=2*(mx*SHARED_SIZE_PRIMARY+my);
	cap_p=2*((mx+1)*SHARED_SIZE_PRIMARY+my);
	ketre=shared_ket[cap];
	ketim=shared_ket[cap+1];
	ketre_p=shared_ket[cap_p];
	ketim_p=shared_ket[cap_p+1];
	// temp1=costx*ket(x,y) - i sintx ket(x+1,y) exp(i ax kx)
	temp1re=costx*ketre+sintx*(ketim_p*coskxax+ketre_p*sinkxax);
	temp1im=costx*ketim-sintx*(ketre_p*coskxax-ketim_p*sinkxax);
	// temp2=- i sintx ket(x,y) exp(-i ax kx) + costx*ket(x+1,y);
	temp2re=sintx*(-ketre*sinkxax+ketim*coskxax) + costx*ketre_p;
	temp2im=sintx*(-ketim*sinkxax-ketre*coskxax) + costx*ketim_p;
	//ket(x,y) = (costx-i*sintx)*temp1
	shared_ket[cap]=costx*temp1re+sintx*temp1im;
	shared_ket[cap+1]=costx*temp1im-sintx*temp1re;
	//ket(x+1,y) = (costx-i*sintx)*temp2
	shared_ket[cap_p]=costx*temp2re+sintx*temp2im;
	shared_ket[cap_p+1]=costx*temp2im-sintx*temp2re;
	__syncthreads();
	if (threadIdx.y < SHARED_SIZE_SECONDARY-1)
	{
		mx = 2*threadIdx.y+1;
		cap=2*(mx*SHARED_SIZE_PRIMARY+my);
		cap_p=2*((mx+1)*SHARED_SIZE_PRIMARY+my);
		ketre=shared_ket[cap];
		ketim=shared_ket[cap+1];
		ketre_p=shared_ket[cap_p];
		ketim_p=shared_ket[cap_p+1];
		temp1re=costx*ketre+sintx*(ketim_p*coskxax+ketre_p*sinkxax);
		temp1im=costx*ketim-sintx*(ketre_p*coskxax-ketim_p*sinkxax);
		temp2re=sintx*(-ketre*sinkxax+ketim*coskxax) + costx*ketre_p;
		temp2im=sintx*(-ketim*sinkxax-ketre*coskxax) + costx*ketim_p;
		shared_ket[cap]=costx*temp1re+sintx*temp1im;
		shared_ket[cap+1]=costx*temp1im-sintx*temp1re;
		shared_ket[cap_p]=costx*temp2re+sintx*temp2im;
		shared_ket[cap_p+1]=costx*temp2im-sintx*temp2re;
	}
	__syncthreads();
	mx = threadIdx.x;
	my = 2*threadIdx.y;
	cap=2*(mx*SHARED_SIZE_PRIMARY+my);
	cap_p=2*(mx*SHARED_SIZE_PRIMARY+(my+1));
	ketre=shared_ket[cap];
	ketim=shared_ket[cap+1];
	ketre_p=shared_ket[cap_p];
	ketim_p=shared_ket[cap_p+1];
	temp1re=costy*ketre+sinty*(ketim_p*coskyay+ketre_p*sinkyay);
	temp1im=costy*ketim-sinty*(ketre_p*coskyay-ketim_p*sinkyay);
	temp2re=sinty*(-ketre*sinkyay+ketim*coskyay) + costy*ketre_p;
	temp2im=sinty*(-ketim*sinkyay-ketre*coskyay) + costy*ketim_p;
	shared_ket[cap]=costy*temp1re+sinty*temp1im;
	shared_ket[cap+1]=costy*temp1im-sinty*temp1re;
	shared_ket[cap_p]=costy*temp2re+sinty*temp2im;
	shared_ket[cap_p+1]=costy*temp2im-sinty*temp2re;
	__syncthreads();
	if (threadIdx.y < SHARED_SIZE_SECONDARY-1)
	{
		my = 2*threadIdx.y+1;
		cap=2*(mx*SHARED_SIZE_PRIMARY+my);
		cap_p=2*(mx*SHARED_SIZE_PRIMARY+(my+1));
		ketre=shared_ket[cap];
		ketim=shared_ket[cap+1];
		ketre_p=shared_ket[cap_p];
		ketim_p=shared_ket[cap_p+1];
		temp1re=costy*ketre+sinty*(ketim_p*coskyay+ketre_p*sinkyay);
		temp1im=costy*ketim-sinty*(ketre_p*coskyay-ketim_p*sinkyay);
		temp2re=sinty*(-ketre*sinkyay+ketim*coskyay) + costy*ketre_p;
		temp2im=sinty*(-ketim*sinkyay-ketre*coskyay) + costy*ketim_p;
		shared_ket[cap]=costy*temp1re+sinty*temp1im;
		shared_ket[cap+1]=costy*temp1im-sinty*temp1re;
		shared_ket[cap_p]=costy*temp2re+sinty*temp2im;
		shared_ket[cap_p+1]=costy*temp2im-sinty*temp2re;
	}
	__syncthreads();
	for (int id=0; id<2; id++)
	{
		my = 2*threadIdx.y+id;
		rap=2*(((Nx+globalx0+mx)%Nx)*Ny+((globaly0+my+Ny)%Ny));
		cap=2*(mx*SHARED_SIZE_PRIMARY+my);
		ketre=shared_ket[cap];
		ketim=shared_ket[cap+1];
		//tmp=-deltat*(energias[rap]-2.0*Vx-2.0*Vy);
		//costmp=(float) cos(tmp);
		//sintmp=(float) sin(tmp);
		costmp=prepared_energias[rap];
		sintmp=prepared_energias[rap+1];
		kre = ketre*costmp - ketim*sintmp;
		kim = ketim*costmp + ketre*sintmp;
		shared_ket[cap] = kre;
		shared_ket[cap+1] =kim;
	}
	__syncthreads();
	if (threadIdx.y < SHARED_SIZE_SECONDARY-1)
	{
		my = 2*threadIdx.y+1;
		cap=2*(mx*SHARED_SIZE_PRIMARY+my);
		cap_p=2*(mx*SHARED_SIZE_PRIMARY+(my+1));
		ketre=shared_ket[cap];
		ketim=shared_ket[cap+1];
		ketre_p=shared_ket[cap_p];
		ketim_p=shared_ket[cap_p+1];
		temp1re=costy*ketre+sinty*(ketim_p*coskyay+ketre_p*sinkyay);
		temp1im=costy*ketim-sinty*(ketre_p*coskyay-ketim_p*sinkyay);
		temp2re=sinty*(-ketre*sinkyay+ketim*coskyay) + costy*ketre_p;
		temp2im=sinty*(-ketim*sinkyay-ketre*coskyay) + costy*ketim_p;
		shared_ket[cap]=costy*temp1re+sinty*temp1im;
		shared_ket[cap+1]=costy*temp1im-sinty*temp1re;
		shared_ket[cap_p]=costy*temp2re+sinty*temp2im;
		shared_ket[cap_p+1]=costy*temp2im-sinty*temp2re;
	}
	__syncthreads();
	my = 2*threadIdx.y;
	cap=2*(mx*SHARED_SIZE_PRIMARY+my);
	cap_p=2*(mx*SHARED_SIZE_PRIMARY+(my+1));
	ketre=shared_ket[cap];
	ketim=shared_ket[cap+1];
	ketre_p=shared_ket[cap_p];
	ketim_p=shared_ket[cap_p+1];
	temp1re=costy*ketre+sinty*(ketim_p*coskyay+ketre_p*sinkyay);
	temp1im=costy*ketim-sinty*(ketre_p*coskyay-ketim_p*sinkyay);
	temp2re=sinty*(-ketre*sinkyay+ketim*coskyay) + costy*ketre_p;
	temp2im=sinty*(-ketim*sinkyay-ketre*coskyay) + costy*ketim_p;
	shared_ket[cap]=costy*temp1re+sinty*temp1im;
	shared_ket[cap+1]=costy*temp1im-sinty*temp1re;
	shared_ket[cap_p]=costy*temp2re+sinty*temp2im;
	shared_ket[cap_p+1]=costy*temp2im-sinty*temp2re;
	__syncthreads();
	my = threadIdx.x;
	if (threadIdx.y < SHARED_SIZE_SECONDARY-1)
	{
		mx = 2*threadIdx.y+1;
		cap=2*(mx*SHARED_SIZE_PRIMARY+my);
		cap_p=2*((mx+1)*SHARED_SIZE_PRIMARY+my);
		ketre=shared_ket[cap];
		ketim=shared_ket[cap+1];
		ketre_p=shared_ket[cap_p];
		ketim_p=shared_ket[cap_p+1];
		temp1re=costx*ketre+sintx*(ketim_p*coskxax+ketre_p*sinkxax);
		temp1im=costx*ketim-sintx*(ketre_p*coskxax-ketim_p*sinkxax);
		temp2re=sintx*(-ketre*sinkxax+ketim*coskxax) + costx*ketre_p;
		temp2im=sintx*(-ketim*sinkxax-ketre*coskxax) + costx*ketim_p;
		shared_ket[cap]=costx*temp1re+sintx*temp1im;
		shared_ket[cap+1]=costx*temp1im-sintx*temp1re;
		shared_ket[cap_p]=costx*temp2re+sintx*temp2im;
		shared_ket[cap_p+1]=costx*temp2im-sintx*temp2re;
	}
	__syncthreads();
	mx = 2*threadIdx.y;
	cap=2*(mx*SHARED_SIZE_PRIMARY+my);
	cap_p=2*((mx+1)*SHARED_SIZE_PRIMARY+my);
	ketre=shared_ket[cap];
	ketim=shared_ket[cap+1];
	ketre_p=shared_ket[cap_p];
	ketim_p=shared_ket[cap_p+1];
	temp1re=costx*ketre+sintx*(ketim_p*coskxax+ketre_p*sinkxax);
	temp1im=costx*ketim-sintx*(ketre_p*coskxax-ketim_p*sinkxax);
	temp2re=sintx*(-ketre*sinkxax+ketim*coskxax) + costx*ketre_p;
	temp2im=sintx*(-ketim*sinkxax-ketre*coskxax) + costx*ketim_p;
	shared_ket[cap]=costx*temp1re+sintx*temp1im;
	shared_ket[cap+1]=costx*temp1im-sintx*temp1re;
	shared_ket[cap_p]=costx*temp2re+sintx*temp2im;
	shared_ket[cap_p+1]=costx*temp2im-sintx*temp2re;
	__syncthreads();
	
	if (threadIdx.x>=overhead && 2*threadIdx.y>=overhead && threadIdx.x<SHARED_SIZE_PRIMARY-overhead && 2*threadIdx.y<SHARED_SIZE_PRIMARY-overhead-1)
	{
		if ( (globalx0+threadIdx.x < Nx) && (globaly0+2*threadIdx.y<Ny-1) )
		{
			if ( dataketid == 0 )
			{
				ket1[gpos]   = shared_ket[spos];
				ket1[gpos+1] = shared_ket[spos+1];
				ket1[gpos+2] = shared_ket[spos+2];
				ket1[gpos+3] = shared_ket[spos+3];
			} else 
			{
				ket0[gpos]   = shared_ket[spos];
				ket0[gpos+1] = shared_ket[spos+1];
				ket0[gpos+2] = shared_ket[spos+2];
				ket0[gpos+3] = shared_ket[spos+3];
			}
		}
	}
} }}}
__global__ void H2d_x(float *ket, float deltat, int id, int pbcx, int Nx, int Ny, float Vx, float sinkxax, float coskxax) {{{
{
//	int Radius = 3;
//	int BlockDim_x = blockDim.x;
//	__shared__ float s_a[BlockDim_x+2*Radius];
//	int global_ix = blockDim.x*blockIdx.x + threadIdx.x;
//	int local_ix = Radius + threadIdx.x;
//	
//	s_a[local_ix] = input[global_ix];
//	
//	if ( threadIdx.x < Radius )
//	{
//		s_a[local_ix-Radius]     = input[global_ix-Radius];
//		s_a[local_ix+BlockDim_x] = input[global_ix+BlockDim_x];
//	}
//	__syncthreads();
//	
//	float value = 0.0;
//	for (offset = -Radius; offset <=Radius; offset++)
//		value += s_a[local_ix + offset];
//
//	output[global_ix] = value;

//	int globalSizeX = 1000;
//	int globalSizeY = 1000;
//	int RadiusX = 3;
//	int RadiusY = 3;
//	int BlockDimX = blockDim.x;
//	int BlockDimY = blockDim.y;
//	int localSizeX = BlockDimX+2*RadiusX;
//	int localSizeY = BlockDimY+2*RadiusY;
//	__shared__ float s_a[SizeX*SizeY];
//	int global_ix = blockDim.x*blockIdx.x + threadIdx.x;
//	int global_iy = blockDim.y*blockIdx.y + threadIdx.y;
//	int local_ix = RadiusX + threadIdx.x;
//	int local_iy = RadiusY + threadIdx.y;
//	
//	s_a[local_ix*localSizeY+local_iy] = input[global_ix*globalSizeY+global_iy];
//	
//	if ( threadIdx.x < RadiusX )
//	{
//		s_a[(local_ix-RadiusX)*localSizeY+local_iy]   = input[(global_ix-Radius)*globalSizeY+global_iy];
//		s_a[(local_ix+BlockDimX)*localSizeY+local_iy] = input[(global_ix+BlockDim_x)*globalSizeY+global_iy];
//	}
//	if ( threadIdx.y < RadiusY )
//	{
//		s_a[local_ix*localSizeY+local_iy-RadiusY]   = input[global_ix*globalSizeY+global_iy-RadiusY];
//		s_a[local_ix*localSizeY+local_iy+BlockDimY] = input[global_ix*globalSizeY+global_iy+BlockDimY];
//	}
//	__syncthreads();
//	
//	float value = 0.0;
//	for (offsetX = -RadiusX; offsetX <=RadiusX; offsetX++)
//		for (offsetY = -RadiusY; offsetY <=RadiusY; offsetY++)
//			value += s_a[(local_ix + offsetX)*localSizeY+local_iy+offsetY];
//
//	output[global_ix*globalSizeY+global_iy] = value;

/*	
	int globalSizeX = 500;
	int globalSizeY = 500;
	int RadiusX = 3;
	int RadiusY = 3;
	int BlockDimX = blockDim.x;
	int BlockDimY = blockDim.y;
	int localSizeX = BlockDimX+2*RadiusX;
	int localSizeY = BlockDimY+2*RadiusY;
	__shared__ float s_a[SizeX*SizeY];
	int global_ix = blockDim.x*blockIdx.x + threadIdx.x;
	int global_iy = blockDim.y*blockIdx.y + threadIdx.y;
	int local_ix = RadiusX + threadIdx.x;
	int local_iy = RadiusY + threadIdx.y;
	
	s_a[local_ix*localSizeY+local_iy] = input[global_ix*globalSizeY+global_iy];
	
	if ( threadIdx.x < RadiusX )
	{
		s_a[(local_ix-RadiusX)*localSizeY+local_iy]   = input[(global_ix-Radius)*globalSizeY+global_iy];
		s_a[(local_ix+BlockDimX)*localSizeY+local_iy] = input[(global_ix+BlockDim_x)*globalSizeY+global_iy];
	}
	if ( threadIdx.y < RadiusY )
	{
		s_a[local_ix*localSizeY+local_iy-RadiusY]   = input[global_ix*globalSizeY+global_iy-RadiusY];
		s_a[local_ix*localSizeY+local_iy+BlockDimY] = input[global_ix*globalSizeY+global_iy+BlockDimY];
	}
	__syncthreads();
	
	float value = 0.0;
	for (offsetX = -RadiusX; offsetX <=RadiusX; offsetX++)
		for (offsetY = -RadiusY; offsetY <=RadiusY; offsetY++)
			value += s_a[(local_ix + offsetX)*localSizeY+local_iy+offsetY];

	output[global_ix*globalSizeY+global_iy] = value;
*/
	
	int my = blockDim.x * blockIdx.x + threadIdx.x;
	int Nstart;
	int Nfinal;
	if (id==1)
	{
		Nstart=0;
		Nfinal=Nx-1;
	}
	if (id==2)
	{
		Nstart=1;
		Nfinal=Nx-2;
	}
	float cost, sint, temp1re, temp1im, temp2re, temp2im;;
	int mx, cap, cap_p;
	cost=(float) cos(deltat*Vx);
	sint=(float) sin(deltat*Vx);
	float ketre, ketim, ketre_p, ketim_p;
	if (my<Ny)
	{
		
		
		for (mx=Nstart; mx<Nfinal; mx+=2)
		{
			cap=2*(mx*Ny+my);
			cap_p=2*((mx+1)*Ny+my);
			ketre=ket[cap];
			ketim=ket[cap+1];
			ketre_p=ket[cap_p];
			ketim_p=ket[cap_p+1];
			// temp1=cost*ket(x,y) - i sint ket(x+1,y) exp(i ax kx)
			temp1re=cost*ketre+sint*(ketim_p*coskxax+ketre_p*sinkxax);	// checked: ok - fixed one sign error!!
			temp1im=cost*ketim-sint*(ketre_p*coskxax-ketim_p*sinkxax);	// checked: ok - fixed one sign error!!
			// temp2=- i sint ket(x,y) exp(-i ax kx) + cost*ket(x+1,y);
			temp2re=sint*(-ketre*sinkxax+ketim*coskxax) + cost*ketre_p;	// checked: ok
			temp2im=sint*(-ketim*sinkxax-ketre*coskxax) + cost*ketim_p;	// checked: ok
			//ket(x,y) = (cost-i*sint)*temp1
			ket[cap]=cost*temp1re+sint*temp1im;							// checked: ok
			ket[cap+1]=cost*temp1im-sint*temp1re;						// checked: ok
			//ket(x+1,y) = (cost-i*sint)*temp2
			ket[cap_p]=cost*temp2re+sint*temp2im;						// checked: ok
			ket[cap_p+1]=cost*temp2im-sint*temp2re;					// checked: ok
		}
	}
	if ((pbcx)&&(id==2))
	{
		if (my<Ny)
		{
			cap=2*((Nx-1)*Ny+my);
			cap_p=2*my;
			ketre=ket[cap];
			ketim=ket[cap+1];
			ketre_p=ket[cap_p];
			ketim_p=ket[cap_p+1];
			temp1re=cost*ketre+sint*(ketim_p*coskxax+ketre_p*sinkxax);	// checked: ok - fixed one sign error!!
			temp1im=cost*ketim-sint*(ketre_p*coskxax-ketim_p*sinkxax);	// checked: ok - fixed one sign error!!
			temp2re=sint*(-ketre*sinkxax+ketim*coskxax) + cost*ketre_p;	// checked: ok
			temp2im=sint*(-ketim*sinkxax-ketre*coskxax) + cost*ketim_p;	// checked: ok
			ket[cap]=cost*temp1re+sint*temp1im;							// checked: ok
			ket[cap+1]=cost*temp1im-sint*temp1re;						// checked: ok
			ket[cap_p]=cost*temp2re+sint*temp2im;						// checked: ok
			ket[cap_p+1]=cost*temp2im-sint*temp2re;					// checked: ok
		}
	}
} }}}
__global__ void H2d_y(float *ket, float deltat, int id, int pbcy, int Nx, int Ny, float Vy, float sinkyay, float coskyay) {{{
{
	int mx = blockDim.x * blockIdx.x + threadIdx.x;
	int Nstart;
	int Nfinal;
	if (id==1)
	{
		Nstart=0;
		Nfinal=Ny-1;
	}
	if (id==2)
	{
		Nstart=1;
		Nfinal=Ny-2;
	}
	float cost, sint, temp1re, temp1im, temp2re, temp2im;;
	int  my, cap, cap_p;
	cost=(float) cos(deltat*Vy);
	sint=(float) sin(deltat*Vy);
	float ketre, ketim, ketre_p, ketim_p;
	if (mx<Nx)
	{
		for (my=Nstart; my<Nfinal; my+=2)
		{
			cap=2*(mx*Ny+my);
			cap_p=2*(mx*Ny+(my+1));
			ketre=ket[cap];
			ketim=ket[cap+1];
			ketre_p=ket[cap_p];
			ketim_p=ket[cap_p+1];
			temp1re=cost*ketre+sint*(ketim_p*coskyay+ketre_p*sinkyay);	// checked: ok - fixed one sign error!!
			temp1im=cost*ketim-sint*(ketre_p*coskyay-ketim_p*sinkyay);	// checked: ok - fixed one sign error!!
			temp2re=sint*(-ketre*sinkyay+ketim*coskyay) + cost*ketre_p;	// checked: ok
			temp2im=sint*(-ketim*sinkyay-ketre*coskyay) + cost*ketim_p;	// checked: ok
			ket[cap]=cost*temp1re+sint*temp1im;							// checked: ok
			ket[cap+1]=cost*temp1im-sint*temp1re;						// checked: ok
			ket[cap_p]=cost*temp2re+sint*temp2im;						// checked: ok
			ket[cap_p+1]=cost*temp2im-sint*temp2re;					// checked: ok
		}
	}
	if ((pbcy)&&(id==2))
	{
		if (mx<Nx)
		{
			cap=2*(mx*Ny+Ny-1);
			cap_p=2*(mx*Ny);
			ketre=ket[cap];
			ketim=ket[cap+1];
			ketre_p=ket[cap_p];
			ketim_p=ket[cap_p+1];
			temp1re=cost*ketre+sint*(ketim_p*coskyay+ketre_p*sinkyay);	// checked: ok - fixed one sign error!!
			temp1im=cost*ketim-sint*(ketre_p*coskyay-ketim_p*sinkyay);	// checked: ok - fixed one sign error!!
			temp2re=sint*(-ketre*sinkyay+ketim*coskyay) + cost*ketre_p;	// checked: ok
			temp2im=sint*(-ketim*sinkyay-ketre*coskyay) + cost*ketim_p;	// checked: ok
			ket[cap]=cost*temp1re+sint*temp1im;							// checked: ok
			ket[cap+1]=cost*temp1im-sint*temp1re;						// checked: ok
			ket[cap_p]=cost*temp2re+sint*temp2im;						// checked: ok
			ket[cap_p+1]=cost*temp2im-sint*temp2re;					// checked: ok
		}
	}
} }}}
__global__ void Hdiag2d(float *ket, float deltat, float *energias, int Nx, int Ny, float Vx, float Vy) {{{
{
	int mx = blockDim.x * blockIdx.x + threadIdx.x;
	int my;
	int cap;
	int rap;
	float kre, kim, tmp, costmp, sintmp;
	float ketre, ketim;
	if (mx < Nx)
	{
		for (my=0; my<Ny; my++)
		{
			rap=mx*Ny+my;
			cap=2*rap;
			ketre=ket[cap];
			ketim=ket[cap+1];
			tmp=-deltat*(energias[rap]-2.0*Vx-2.0*Vy);
			costmp=(float) cos(tmp);
			sintmp=(float) sin(tmp);
			kre = ketre*costmp - ketim*sintmp;
			kim = ketim*costmp + ketre*sintmp;
			ket[cap] = kre;
			ket[cap+1] =kim;
		}
	}

} }}}
// } } }
// commented main 
int main(int argc, char *argv[])
{
 // Definition of variables {{{ 
	int		debug	=	0;
	float	pi		= 4.0*((float) atan(1.0));
	float	hbar	= 1.0; // XXX XXX XXX WE SET hbar = 1.0 XXX XXX XXX
	int		Nx		= 400;
	int		Ny		= 400;
	float	dNx		= (float) Nx;
	float	dNy		= (float) Ny;
	float	Lx		=  20.0;
	float	Ly		=  20.0;
	int		pbcx	=   1; // C has no boolean data type; we use false=0, true=1
	int		pbcy	=   1; // C has no boolean data type; we use false=0, true=1
	float	lambda	=   0.0; // XXX NEVER USED XXX
	float	innerR	=   0.0; // XXX NEVER USED XXX
	float	outerR	=   4.0; // XXX NEVER USED XXX
	float	alpha	= 100.0;
	float	beta	=  20.0;
	float	x0		= -12.0;
	float	y0		=   0.0; // XXX NEVER USED XXX
	float	sx		=   3.0;
	float	sy		=   3.0; // XXX NEVER USED XXX
	float	px0		=   0.0;	// XXX formerly: 0.1*pi*hbar/ax;
	float	py0		=   0.0;
	float	dt		=   0.001;
//	float	dtmax	=	1.0; // XXX NEVER USED XXX
	int		order	=	2; // XXX NEVER USED XXX
//	int		gsteps	= 100; // XXX NEVER USED XXX
	char	*file	= "test"; // XXX NEVER USED XXX
//	char	*filein	= "test"; // XXX NEVER USED XXX
	char	*dir	= "data"; // XXX NEVER USED XXX
	int		verbose	=	0;
//	int		debut	=	0; // XXX NEVER USED XXX
//	int		ground	=	0; // XXX NEVER USED XXX
//	int		nodyn	=	0; // XXX NEVER USED XXX
	char	*form	= "DT"; // XXX NEVER USED XXX
	int		pars	=	0; // XXX NEVER USED XXX
	int 	abort	=	0; // XXX NEVER USED XXX
	float	cloakEnergy = 0.0; // XXX NEVER USED XXX
	float	mass0=0.5;
	float	tmax	=	10.0;

	char	*filepars;
	float	ax;
	float	ay;
//}}}
// Command line arguments analysis {{{
	int n=1;
	while (n<argc)
	{
		if (strcmp(argv[n],"-pars")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-pars found at position %i, argument is %s\n", n, argv[n+1]);
				filepars=argv[n+1]; 
				pars=1;
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-nx")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-nx found at position %i, argument is %s\n", n, argv[n+1]);
				Nx=atoi(argv[n+1]);
				dNx=(float) Nx;
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-ny")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-ny found at position %i, argument is %s\n", n, argv[n+1]);
				Ny=atoi(argv[n+1]); 
				dNy=(float) Ny;
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-Lx")==0)
		{
			if (n+1<argc) 
			{
				if (debug) printf("-Lx found at position %i, argument is %s\n", n, argv[n+1]);
				Lx=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-Ly")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-Ly found at position %i, argument is %s\n", n, argv[n+1]);
				Ly=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-lambda")==0)
		{
			if (n+1<argc) 
			{
				if (debug) printf("-lambda found at position %i, argument is %s\n", n, argv[n+1]);
				lambda=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-R1")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-R1 found at position %i, argument is %s\n", n, argv[n+1]);
				innerR=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-R2")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-R2 found at position %i, argument is %s\n", n, argv[n+1]);
				outerR=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-target")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-target found at position %i, argument is %s\n", n, argv[n+1]);
				cloakEnergy=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-alpha")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-alpha found at position %i, argument is %s\n", n, argv[n+1]);
				alpha=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-beta")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-beta found at position %i, argument is %s\n", n, argv[n+1]);
				beta=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-x0")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-x0 found at position %i, argument is %s\n", n, argv[n+1]);
				x0=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-y0")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-y0 found at position %i, argument is %s\n", n, argv[n+1]);
				y0=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-sx")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-sx found at position %i, argument is %s\n", n, argv[n+1]);
				sx=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-sy")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-sy found at position %i, argument is %s\n", n, argv[n+1]);
				sy=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-px0")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-px0 found at position %i, argument is %s\n", n, argv[n+1]);
				px0=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-py0")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-py0 found at position %i, argument is %s\n", n, argv[n+1]);
				py0=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-dt")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-dt found at position %i, argument is %s\n", n, argv[n+1]);
				dt=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-order")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-order found at position %i, argument is %s\n", n, argv[n+1]);
				order=atoi(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-time")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-time found at position %i, argument is %s\n", n, argv[n+1]);
				tmax=(float) atof(argv[n+1]);
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-file")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-file found at position %i, argument is %s\n", n, argv[n+1]);
				file=argv[n+1];
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-format")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-format found at position %i, argument is %s\n", n, argv[n+1]);
				form=argv[n+1];
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-dir")==0)
		{
			if (n+1<argc)
			{
				if (debug) printf("-dir found at position %i, argument is %s\n", n, argv[n+1]);
				dir=argv[n+1];
				n+=2;
			}
		} else
		if (strcmp(argv[n],"-V")==0)
		{
			if (debug) printf("-verbose found at position %i\n", n);
			verbose=1;
			n++;
		} else
		if (strcmp(argv[n],"-pbcx")==0)
		{
			if (debug) printf("-pbcx found at position %i\n", n);
			pbcx=1;
			n++;
		} else
		if (strcmp(argv[n],"-pbcy")==0)
		{
			if (debug) printf("-pbcy found at position %i\n", n);
			pbcy=1;
			n++;
		} else
		if (strcmp(argv[n],"-debug")==0)
		{
			debug=1;
			printf("-debug found at position %i; all subsequent arguments are listed\n", n);
			n++;
		} else
		{
			printf("ERROR: Wrong argument at position %i: $s\n", n, argv[n]);
			printf("Cloaking of matter waves in 2D. Wrong parameters.\n");
			printf("Options: \n");
			printf("    System:\n");
			printf("        -nx -ny  Points in X and Y direction\n");
			printf("        -Lx -Ly  Size in X and Y direction (system is from -L to +L)\n");
			printf("        -lambda  Boson-boson interaction strength\n");
			printf("        -R1 -R2  Internal and external cloak radius\n");
			printf("        -target  target energy (factor from chemical potential)\n");
			printf("        -alpha   Intensity of perturbation at origin\n");
			printf("        -beta    1/Radius^2 of perturbation at origin\n");
			printf("        -pbcx -pbcy Periodic boundary conditions\n");
			printf("    Wave packet:\n"         );
			printf("        -x0 -y0  Initial position\n");
			printf("        -sx -sy  Position dispersion (sigmas)\n");
			printf("        -px0 -py0  Initial momentum\n");
			printf("    Simulation:\n"         );
			printf("        -time   Evolution time\n");
			printf("        -dt      Time step\n");
			printf("        -order   Algorithm order (default 2)\n");
			printf("    Input/Output:\n"         );
			printf("        -file    Output file base name\n");
			printf("        -format  Output file format (DT, TP)\n");
			printf("        -pars    Parameters file (all other input is ignored)\n");
			printf("        -dir     Input/Output directory\n");
			printf("        -V       Verbose\n");
			printf("        -debug   Debug\n");
			abort = 1;
			exit(1);
		}
	}
//}}}	
// Memory allocation {{{
	// XXX	file=trim(adjustl(dir))//'/'//trim(file)
	// XXX	fileini=trim(adjustl(dir))//'/'//trim(fileini)
	//int		steps	= (int) (tmax/dt);
	if (pbcx) ax=2.0*Lx/dNx; else ax=2.0*Lx/(dNx+1.0);
	if (pbcy) ay=2.0*Ly/dNy; else ay=2.0*Ly/(dNy+1.0);
	float	Vx		= -hbar*hbar/(2.0*ax*ax);
	float	Vy		= -hbar*hbar/(2.0*ay*ay);
	float	kx		=	0.0;	// rotx*pi/Lx	
	float	ky		=	0.0;	// roty*pi/Ly	
	
	if (debug) printf("Allocating memory\n");
	int MatrixSize=Nx*Ny;
	float *phi0 = (float*) malloc(2*MatrixSize*sizeof(float));
	if (phi0==0)
	{
		printf("ERROR: memory of variable phi0 could not be allocated.\n");
		exit(1);
	}
	float *phit = (float*) malloc(2*MatrixSize*sizeof(float));
	if (phit==0)
	{
		printf("ERROR: memory of variable phit could not be allocated.\n");
		exit(1);
	}
	float *potential = (float*) malloc(MatrixSize*sizeof(float));
	if (potential==0)
	{
		printf("ERROR: memory of variable potential could not be allocated.\n");
		exit(1);
	}
	float *Et = (float*) malloc(MatrixSize*sizeof(float));
	if (Et==0)
	{
		printf("ERROR: memory of variable Et could not be allocated.\n");
		exit(1);
	}
	float *pert = (float*) malloc(MatrixSize*sizeof(float));
	if (pert==0)
	{
		printf("ERROR: memory of variable pert could not be allocated.\n");
		exit(1);
	}
//}}}
// Compute initial conditions{{{
	int mx, my, carraypos, rarraypos;
	float x, y, x2, y2, dx, dx2, sx2, exp_local, exp_xplus, exp_xminus, r2;
	sx2=sx*sx;
	for (mx=0; mx<Nx; mx++)
	{
		x=((float)mx)*ax-Lx;
		x2=x*x;
		dx=x-x0;
		dx2=dx*dx;
		exp_local  = ((float) exp(-0.5*dx2/sx2));
		exp_xminus = ((float) exp(-0.5*(dx-2.0*Lx)*(dx-2.0*Lx)/sx2));
		exp_xplus  = ((float) exp(-0.5*(dx+2.0*Lx)*(dx+2.0*Lx)/sx2));
		for (my=0; my<Ny; my++)
		{
			y=((float)my)*ay-Ly;
			y2=y*y;
			rarraypos=mx*Ny+my;
			carraypos=2*(mx*Ny+my);
			float phi0re=0.0;
			float phi0im=0.0;
			phi0re = exp_local*((float) cos(px0*x+py0*y));
			phi0im = exp_local*((float) sin(px0*x+py0*y));
			if (pbcx)
			{
				phi0re += exp_xminus * ((float) cos(px0*(x-2.0*Lx)+py0*y));
				phi0re += exp_xplus  * ((float) cos(px0*(x+2.0*Lx)+py0*y));
				phi0im += exp_xminus * ((float) sin(px0*(x-2.0*Lx)+py0*y));
				phi0im += exp_xplus  * ((float) sin(px0*(x+2.0*Lx)+py0*y));
			}
			if (pbcy)
			{
//				phi0re += exp_local * ((float) cos(px0*x+py0*(y-2.0*Ly)));
//				phi0re += exp_local * ((float) cos(px0*x+py0*(y+2.0*Ly)));
//				phi0im += exp_local * ((float) sin(px0*x+py0*(y-2.0*Ly)));
//				phi0im += exp_local * ((float) sin(px0*x+py0*(y-2.0*Ly)));
			}
			phi0[carraypos]   = phi0re;
			phi0[carraypos+1] = phi0im;
			r2=x2+y2;
			if (r2<beta*beta) potential[rarraypos]=alpha;
			//potential[rarraypos]  = alpha*exp(-beta*r2);
		}
	}
	float rtemp=0.0;
	float phi0re, phi0re2, phi0im, phi0im2;
	for (mx=0; mx<Nx; mx++)
	{
		for (my=0; my<Ny; my++)
		{
			carraypos=2*(mx*Ny+my);
			phi0re=phi0[carraypos];
			phi0re2=phi0re*phi0re;
			phi0im=phi0[carraypos+1];
			phi0im2=phi0im*phi0im;
			rtemp += ax*ay*(phi0re2+phi0im2);
		}
	}
	float sqr_rtemp=sqrt(rtemp);
	for (mx=0; mx<Nx; mx++)
	{
		for (my=0; my<Ny; my++)
		{
			carraypos=2*(mx*Ny+my);
			phi0[carraypos]   *= 1.0/sqr_rtemp;
			phi0[carraypos+1] *= 1.0/sqr_rtemp;
		}
	}
	// Calculate chemical potential and <r^2>
	float ctempre = 0.0;
	float ctempim = 0.0;
	rtemp = 0.0;
	if (verbose) printf("Computing chemical potential\n");
	int mxm, mxp, mym, myp;
	float rxm, rxp, rym, ryp;
	float prefact, cVx, cVy;
	for (mx=1; mx<(Nx-1); mx++)
	{
		x=((float)mx)*ax-Lx;
		mxp = mx+1;
		if ((pbcx)&&(mx==Nx)) mxp=1;
		rxp=((float)mxp)*ax-Lx;
		mxm = mx-1;
		if ((pbcx)&&(mx==1)) mxm=Nx;
		rxm=((float)mxm)*ax-Lx;
		for (my=1; my<(Ny-1); my++)
		{
			y=((float)my)*ay-Ly;
			myp = my+1;
			if ((pbcy)&&(my==Ny)) myp=1;
			ryp=((float)myp)*ay-Ly;
			mym = my-1;
			if ((pbcy)&&(my==1)) mym=Ny;
			rym=((float)mym)*ay-Ly;
			
			carraypos=2*(mx*Ny+my);
			prefact=ax*ay;
// XXX XXX XXX	cVx=Vx*(phi0[2*(mxp*Ny+my)]*massxx[rxp*Ny+y]+phi0[2*(mxm*Ny+my)]*massxx[rxm*Ny+y]-phi0[2*(mx*Ny+my)]*(massxx[rxm*Ny+y]+massxx[rxp*Ny+y]))
			cVx=Vx*(phi0[2*(mxp*Ny+my)]*mass0+phi0[2*(mxm*Ny+my)]*mass0-phi0[2*(mx*Ny+my)]*(mass0+mass0));
// XXX XXX XXX	cVy=Vy*(phi0[2*(mx*Ny+myp)]*massyy[x*Ny+ryp]+phi0[2*(mx*Ny+mym)]*massyy[x*Ny+rym]-phi0[2*(mx*Ny+my)]*(massyy[x*Ny+ryp]+massyy[x*Ny+rym]))
			cVy=Vy*(phi0[2*(mx*Ny+myp)]*mass0+phi0[2*(mx*Ny+mym)]*mass0-phi0[2*(mx*Ny+my)]*(mass0+mass0));
			phi0re=phi0[carraypos];
			phi0im=phi0[carraypos+1];
			ctempre	+=	 prefact*phi0re*(cVx+cVy);
			ctempim	+=	-prefact*phi0im*(cVx+cVy);
		}
	}
	printf("mu = %f + %f*i; taking real part only\n", ctempre, ctempim);
	float mu=ctempre;
	printf("Chemical potential: %f\n", mu);
	float p2=(px0*px0+py0*py0)/(2.0*mass0);
	printf("p^2: %f\n", p2);
	
	rtemp=2.0*Vx*(1.0-((float) cos(px0*ax)))/mass0+2.0*Vy*(1.0-((float) cos(py0*ay)))/mass0;	//Energy of wave
	printf("Lattice energy\n");
	for (mx=0; mx<Nx; mx++)
	{
		x=((float)mx)*ax-Lx;
		//x2=x*x;
		for (my=0; my<Ny; my++)
		{
			y=((float)my)*ay-Ly;
			//y2=y*y;
			//r=sqrt(x2+y2);
			// XXX XXX XXX potential[mx*Ny+my]+=(1.0d0-detLame(1.0d0*mx,1.0d0*my))*(rtemp*cloakEnergy);
		}
	}
	printf("Hoppings: Vx=%f, Vy=%f\n", Vx, Vy);
	if (verbose) printf("Hoppings: Vx=%f, Vy=%f\n", Vx, Vy);
	if (verbose) printf("Momenta:  kx=%f, ky=%f\n", kx, ky);
	// XXX NOT USED XXX if (verbose) printf("Imaginary time step=%f\n", isteps);
	
	if (verbose)
	{
		rtemp = 0.0;
		for (mx=1; mx<Nx; mx++)
		{
			for (my=1; my<Ny; my++)
			{
				carraypos=2*(mx*Ny+my);
				phi0re=phi0[carraypos];
				phi0im=phi0[carraypos+1];
				phi0re2=phi0re*phi0re;
				phi0im2=phi0im*phi0im;
				rtemp+=ax*ay*(phi0re2+phi0im2);
			}
		}
		printf("Initial state module: rtemp=%f\n", rtemp);
	}
//}}}
// Perform imaginary time evolution	{{ {
	int m;
	for (m=0; m<2*Nx*Ny; m++) phit[m] = phi0[m];
	for (m=0; m<Nx*Ny; m++) Et[m] = potential[m];
	if (debug)
	{
		printf("Saving potential [not implemented yet]\n");
	}
	if (debug)
	{
		printf("Saving initial configuration [not implemented yet]\n");
	}
	printf("tmax   %f\n", tmax);
	printf("dt     %f\n", dt);
	printf("order  %d\n",order);
	
	float sinkxax = ((float) sin(kx*ax));
	float coskxax = ((float) cos(kx*ax));
	float sinkyay = ((float) sin(ky*ay));
	float coskyay = ((float) cos(ky*ay));
	FILE *fp;
	fp=fopen("GPU_Suzuki.ket", "w+");
	int Itot, q, irc;
	Itot=100;
	int NbrOfIterations=Itot+1;
	irc = fwrite(&NbrOfIterations, sizeof(int), 1, fp);
	irc = fwrite(&Nx, sizeof(int), 1, fp);
	irc = fwrite(&Ny, sizeof(int), 1, fp);
	irc = fwrite(phit, sizeof(float), 2*Nx*Ny, fp);
	for (q=0; q<Itot; q++)
	{
		printf("%d\n", q);
		evolve2d(phit, tmax, dt, Et, order, Nx, Ny, Vx, Vy, pbcx, pbcy, sinkxax, coskxax, sinkyay, coskyay);
		irc = fwrite(phit, sizeof(float), 2*Nx*Ny, fp);
	}
	fclose(fp);
// }} }
// Free memory and exit {{{
	if (debug) printf("Freeing allocated memory\n");
	free(phi0);
	free(phit);
	free(potential);
	free(Et);
	free(pert);
	if (debug) printf("All done\n");
	return 0;
//}}}
} 

