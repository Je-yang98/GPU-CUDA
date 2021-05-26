__inline__ __device__ void conservativeCirclesCheck
(float minX, float maxX, float minY, float maxY, size_t circleIdx, size_t tIdx, size_t numCircles, uint* inSection) {
    if (circleIdx >= numCircles){
        inSection[tIdx] = 0;
    } 
    else {
        float3 pos = *(float3*)(&cuConstRendererParams.position[circleIdx * 3]);
        float rad = cuConstRendererParams.radius[circleIdx];

        inSection[tIdx] = static_cast<uint> (circleInBoxConservative(pos.x, pos.y, rad, minX, maxX, maxY, minY));
    }
}


__inline__ __device__ void conservativeCirclesRender
(size_t tIdx, size_t circleIdx, uint* inclusiveOutput, uint* probableCircles) {
    if (tIdx == 0){
        if (inclusiveOutput[0] == 1){ 
            probableCircles[0] = circleIdx;
        }    
    } 
    else if (inclusiveOutput[tIdx] == (inclusiveOutput[tIdx-1]+1)){
        probableCircles[inclusiveOutput[tIdx-1]] = circleIdx;
    }
}


__inline__ __device__ void definiteCirclesCheck
(float minX, float maxX, float minY, float maxY, uint circleIdx, size_t tIdx, uint* inSection) {
    float3 pos = *(float3*)(&cuConstRendererParams.position[circleIdx * 3]);
    float rad = cuConstRendererParams.radius[circleIdx];
    inSection[tIdx] = static_cast<uint> (circleInBox(pos.x, pos.y, rad, minX, maxX, maxY, minY));
}


__inline__ __device__ void definiteCirclesRender
(size_t tIdx, uint* inclusiveOutput, uint* definiteCircles, uint* probableCircles) {
    if (tIdx == 0){
        if (inclusiveOutput[0] == 1){
            definiteCircles[0] = probableCircles[0];
        }
    } 
    else if (inclusiveOutput[tIdx] == (inclusiveOutput[tIdx-1]+1)){
        definiteCircles[inclusiveOutput[tIdx-1]] = probableCircles[tIdx];
    }
}


__inline__ __device__ void sharedMemInclusiveScan
(int threadIndex, uint* sInput, uint* sOutput, volatile uint* sScratch, uint size) {
    if (size > WARP_SIZE) {
        uint idata = sInput[threadIndex];
        uint warpResult = warpScanInclusive(threadIndex, idata, sScratch, WARP_SIZE);
        __syncthreads();

        if ((threadIndex & (WARP_SIZE - 1)) == (WARP_SIZE - 1)){
            sScratch[threadIndex >> LOG2_WARP_SIZE] = warpResult;
        }
        __syncthreads();

        if (threadIndex < (SCAN_BLOCK_DIM / WARP_SIZE)) {
            uint val = sScratch[threadIndex];
            sScratch[threadIndex] = warpScanExclusive(threadIndex, val, sScratch, size >> LOG2_WARP_SIZE);
        }
        __syncthreads();

        sOutput[threadIndex] = warpResult + sScratch[threadIndex >> LOG2_WARP_SIZE];

    } 
    else if (threadIndex < WARP_SIZE) {
        uint idata = sInput[threadIndex];
        sOutput[threadIndex] = warpScanInclusive(threadIndex, idata, sScratch, size);
    }
}

__global__ void kernelRenderCircles() {
    size_t tIdx = blockDim.x * threadIdx.y + threadIdx.x;
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    float minX = static_cast<float>(blockIdx.x) / gridDim.x;
    float maxX = minX + static_cast<float>(blockDim.x) / imageWidth;
    float minY = static_cast<float>(blockIdx.y) / gridDim.y;
    float maxY = minY + static_cast<float>(blockDim.y) / imageHeight;

    __shared__ uint inSection[BLOCKSIZE];
    __shared__ uint inclusiveOutput[BLOCKSIZE];
    __shared__ uint probableCircles[BLOCKSIZE];
    __shared__ uint scratchPad[2*BLOCKSIZE];
    
    float4* imgPtr;
    float4 color;
    float2 pixelCenterNorm;

    if (pixelX < imageWidth && pixelY < imageHeight) {
        imgPtr = (float4*) &cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)];
        color = *imgPtr;
        pixelCenterNorm = make_float2(invWidth*(static_cast<float>(pixelX) + 0.5f), invHeight*(static_cast<float>(pixelY) + 0.5f));
    }

    const size_t numCircles = cuConstRendererParams.numCircles;
    
    for (size_t circleIdxStart = 0; circleIdxStart < numCircles; circleIdxStart += BLOCKSIZE) {
        size_t circleIdx = circleIdxStart + tIdx;

        // Find the all circles in the section
        conservativeCirclesCheck(minX, maxX, minY, maxY, circleIdx, tIdx, numCircles, inSection);
        __syncthreads();
        sharedMemInclusiveScan(tIdx, inSection, inclusiveOutput, scratchPad, BLOCKSIZE);
        __syncthreads();
        conservativeCirclesRender(tIdx, circleIdx, inclusiveOutput, probableCircles);
        __syncthreads();

        size_t numConservativeCircles = inclusiveOutput[BLOCKSIZE-1];

        // Find the final circles in the section
        if (tIdx < numConservativeCircles) {
            definiteCirclesCheck(minX, maxX, minY, maxY, probableCircles[tIdx], tIdx, inSection);
        }
        else {
            inSection[tIdx] = 0;
        }
        __syncthreads();
        sharedMemInclusiveScan(tIdx, inSection, inclusiveOutput, scratchPad, BLOCKSIZE);
        __syncthreads();
        uint* definiteCircles = inSection;
        definiteCirclesRender(tIdx, inclusiveOutput, definiteCircles, probableCircles);
        __syncthreads();

        size_t numDefiniteCircles = inclusiveOutput[numConservativeCircles-1];

        // check if pixel is within image
        if (pixelX < imageWidth && pixelY < imageHeight) {
            for (size_t i=0; i<numDefiniteCircles; i++) {
                size_t circleIdx = definiteCircles[i];
                float3 pos = *(float3*)(&cuConstRendererParams.position[circleIdx * 3]);
                shadePixel(circleIdx, pixelCenterNorm, pos, &color);
            }
        }
        __syncthreads();
    }
    if (pixelX < imageWidth && pixelY < imageHeight) {
        *imgPtr = color;
    }

}


__global__ void kernelRenderOneCircle
(short screenMinX, short screenMaxX, short screenMinY, short screenMaxY, float invWidth, float invHeight, int circleIdx){
    int tIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixel = (screenMaxX - screenMinX) * (screenMaxY - screenMinY);

    if (tIndex <= totalPixel){
        int dimX = screenMaxX - screenMinX;
        int pixelX = tIndex % dimX + screenMinX;
        int pixelY = tIndex / dimX + screenMinY;

        short imageWidth = cuConstRendererParams.imageWidth;

        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4*(imageWidth*pixelY + pixelX)]);
        float2 pixelCenterNorm = make_float2(invWidth*(static_cast<float>(pixelX) + 0.5f), invHeight*(static_cast<float>(pixelY) + 0.5f));
        float3 pos = *(float3*)(&cuConstRendererParams.position[circleIdx*3]);
        shadePixel(circleIdx, pixelCenterNorm, pos, imgPtr);
    }
}


void
CudaRenderer::render() {
    dim3 blockDim(BLOCKDIM, BLOCKDIM);
    size_t gridDimX = (image->width + blockDim.x - 1) / blockDim.x;
    size_t gridDimY = (image->height + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridDimX, gridDimY);

    if (numCircles < 7){
        int imageWidth = image->width;
        int imageHeight = image->height;
        float invWidth = 1.f / imageWidth;
        float invHeight = 1.f / imageHeight;
        
        for (int i=0; i<numCircles; i++){
            float px = position[3*i];
            float py = position[3*i + 1];
            float rad = radius[i];

            short minX = static_cast<short>(imageWidth * (px - rad));
            short maxX = static_cast<short>(imageWidth * (px + rad)) + 1;
            short minY = static_cast<short>(imageHeight * (py - rad));
            short maxY = static_cast<short>(imageHeight * (py + rad)) + 1;

            short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
            short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
            short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
            short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;
            
            int totalPixel = (screenMaxX-screenMinX) * (screenMaxY-screenMinY);
            int THREADS_PER_BLOCK = 64;
            int num_blocks = (totalPixel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            kernelRenderOneCircle<<<num_blocks, THREADS_PER_BLOCK>>>(screenMinX, screenMaxX, screenMinY, screenMaxY, invWidth, invHeight, i);
            cudaDeviceSynchronize();
        }
    }
    else {
        kernelRenderCircles<<<gridDim, blockDim>>>();
    } 
    cudaDeviceSynchronize();

}
