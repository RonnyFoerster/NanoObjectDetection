//rename("Raw_Image.tif")
id = getImageID();
title = getTitle();

selectImage(id);

run("Z Project...", "projection=Median");
rename("Median_Background.tif");

imageCalculator("Subtract create 32-bit stack", title,"Median_Background.tif");
rename("Background-Subtracted.tif");

run("Duplicate...", "duplicate");
rename("Before SNR.tif");
//selectWindow("Result of Raw_Image.tif");

run("Gaussian Blur...", "sigma=1 stack");
rename("SNR-Enhanced-Image.tif");