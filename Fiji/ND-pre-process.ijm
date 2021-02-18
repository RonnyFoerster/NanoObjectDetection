rename("Raw_Image.tif")
run("Z Project...", "projection=Median");
rename("Median_Background.tif");
imageCalculator("Subtract create 32-bit stack", "Raw_Image.tif","Median_Background.tif");
selectWindow("Result of Raw_Image.tif");
run("Gaussian Blur...", "sigma=1 stack");
rename("SNR-Enhanced-Image.tif");