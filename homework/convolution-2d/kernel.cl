
__kernel void convolution2D(
    __global float * inputData, __global float * outputData, __constant float * maskData,
    int width, int height, int maskWidth,  int imageChannels){
    //@@ Insert code to implement matrix multiplication here

    int input_row = get_global_id(0); //i
    int input_col = get_global_id(1); //j
    if (input_row > height || input_col > width) { 
        return;
    }
    int maskRadius = maskWidth/2;
    
    for (int k=0; k < imageChannels; k++) {
        float accum = 0.0f; 
        for (int y = -maskRadius; y <= maskRadius; y++) {
            for (int x = -maskRadius; x <= maskRadius; x++) {
                // Offsets for accessing image and kernel data
                int xOffset = input_col + x;
                int yOffset = input_row + y;

                if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height) {
                    float imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];
                    float maskValue = maskData[(y + maskRadius) * maskWidth + (x + maskRadius)];
                    accum += imagePixel * maskValue;
                }
            }
        }
        if (accum < 0) {
            accum = 0; 
        }
        else if (accum > 1) { 
            accum = 1 ;
        }
        outputData[(input_row * width + input_col)*imageChannels + k] = accum;
    }
}
 /**
    maskRadius := maskWidth/2 # this is integer division, so the result is 2
    for i from 0 to height do
        for j from 0 to width do
            for k from 0 to channels
                accum := 0
                for y from -maskRadius to maskRadius do
                    for x from -maskRadius to maskRadius do
                    xOffset := j + x
                    yOffset := i + y
                    if xOffset >= 0 && xOffset < width &&
                        yOffset >= 0 && yOffset < height then
                        imagePixel := I[(yOffset * width + xOffset) * channels + k]
                        maskValue := K[(y+maskRadius)*maskWidth+x+maskRadius]
                        accum += imagePixel * maskValue
                    end
                    end
            end
            # pixels are in the range of 0 to 1
            P[(i * width + j)*channels + k] = clamp(accum, 0, 1)
            end
        end
    end */
