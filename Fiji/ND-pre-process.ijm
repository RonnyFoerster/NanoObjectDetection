
run("Z Project...", "projection=Median");
rename("Median_Background.tif");
imageCalculator("Subtract create 32-bit stack", "unbekanntes_NP_100fps_9700us_10xObj.tif","Median_Background.tif");
selectWindow("Result of unbekanntes_NP_100fps_9700us_10xObj.tif");
run("Gaussian Blur...", "sigma=1 stack");
rename("SNR-Enhanced-Image.tif");