//n = getNumber("How many divisions?", 2);
id = getImageID();
title = getTitle();
numbits = bitDepth();
frames = nSlices();

print("numbits: " + numbits);
getLocationAndSize(locX, locY, sizeW, sizeH);
width = getWidth();
height = getHeight();
n = floor(sqrt(width/height));
print(n);
tileWidth = width / n;

//floor otherwise the n part have not equal width
tileWidth = floor(tileWidth)
//print(tileWidth);

makeRectangle(0, 0, n*tileWidth, height);
run("Crop");

//create white line between images
if (numbits == 16){
	newImage("WhiteLine", "16-bit white", tileWidth, 1, frames);
	run("Set...", "value=65535 stack");
}

for (x = 0; x < n; x++) {
	print("x = " + x);
	offsetX = x * width / n;
	//print(offsetX);
	selectImage(id);
	tileTitle = " [" + x + "]";
	run("Duplicate...", "duplicate");
	rename("Img_" + x);
	makeRectangle(offsetX, 0, tileWidth, height);
	run("Crop");

	if (x>0){
		selectImage("WhiteLine");
	 	run("Duplicate...", "duplicate");
	 	rename("InsertWhiteLine");
	 	
	 	run("Combine...", "stack1=InsertWhiteLine stack2=Img_" + x + " combine");
	 	rename("Add");

 		run("Combine...", "stack1=Combined stack2=Add combine");
	 	rename("Combined");
	}
 	rename("Combined");
	
}

selectImage("WhiteLine");
close();
