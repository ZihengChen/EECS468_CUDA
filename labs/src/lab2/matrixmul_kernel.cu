/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification



// with shared memory

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{   
    const int TILEWIDTH = 32;
    const int SUMWIDTH  = M.width;

    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float M_tile[TILEWIDTH][TILEWIDTH];
    __shared__ float N_tile[TILEWIDTH][TILEWIDTH];

    float Pelement = 0;
    
    int npart = int(SUMWIDTH/TILEWIDTH);
    if (SUMWIDTH%TILEWIDTH >0) npart++;

    for(int part = 0; part<npart; part++){

        // copy from globle memory to shared memonry Row<M.height &&  Col<N.width && 
        if ( Row<M.height && (threadIdx.x + part * TILEWIDTH) < SUMWIDTH ) {
            int Melement_idx_forcopy = Row * M.width + (threadIdx.x + part * TILEWIDTH);
            M_tile[threadIdx.y][threadIdx.x] = M.elements[Melement_idx_forcopy];
        }
        else{
            M_tile[threadIdx.y][threadIdx.x] = 0;
        }

    
        if ( Col<N.width && (threadIdx.y + part*TILEWIDTH) < SUMWIDTH ){
            int Nelement_idx_forcopy = Col + N.width * (threadIdx.y + part * TILEWIDTH);
            N_tile[threadIdx.y][threadIdx.x] = N.elements[Nelement_idx_forcopy];
        }
        else{
            N_tile[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();


        // Cacluate partial results && (part * TILEWIDTH + k) < SUMWIDTH 

        for ( int k = 0; (k < TILEWIDTH ) ; ++k){
            Pelement += M_tile[threadIdx.y][k] * N_tile[k][threadIdx.x];
        }

        __syncthreads();
    }
 
    if ( Col < P.width && Row < P.height ){
        P.elements[ Row * P.width + Col ] = Pelement;
    }
    
}    


/*
// without shared mamory
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{   

    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;

    // check if the Pelement is inside P matrix
    bool INRANGE = false;
    if ( (Col < P.width) && (Row < P.height)) {
        INRANGE = true;
    }

    // Calculate Pelement if InRange is True
    if (INRANGE){
        int SUMWIDTH = M.width; // == N.height
        float Pelement = 0;
        for (int k = 0; k < SUMWIDTH; ++k) {
		    float Melement = M.elements[ Row * M.width + k ]; 
		    float Nelement = N.elements[ Col + N.width * k ]; 
		    Pelement += Melement * Nelement;
	    }
        P.elements[ Row * P.width + Col ] = Pelement;
    }   
}
*/

#endif // #ifndef _MATRIXMUL_KERNEL_H_
