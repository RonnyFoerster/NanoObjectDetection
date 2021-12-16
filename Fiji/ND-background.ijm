//rename("Raw_Image.tif")
id = getImageID();
title = getTitle();

run("Z Project...", "projection=Median");
rename("Median_Background.tif");

makeRectangle(0, 0, getWidth(), getHeight());

run("Plot Profile");