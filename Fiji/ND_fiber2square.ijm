//n = getNumber("How many divisions?", 2);
id = getImageID();
title = getTitle();
numbits = bitDepth();
frames = nSlices();
new_title = title + "-squared"

print("title: " + title)
print("New title: " + new_title)
print("numbits: " + numbits);
getLocationAndSize(locX, locY, sizeW, sizeH);
width = getWidth();
height = getHeight();
n = floor(sqrt(width/height));
print("number of tiles: " + n);
tileWidth = width / n;

//floor otherwise the n part have not equal width
tileWidth = floor(tileWidth)
print("tile Width: " + tileWidth)
print("tile Height: " + height)

makeRectangle(0, 0, n*tileWidth, height);
run("Crop");

//cannot handle 32bit somehow
if (numbits == 32){
	run("16-bit");
}


//create white line between images
if (numbits == 16){
	newImage("WhiteLine", "16-bit white", tileWidth, 1, frames);
	run("Set...", "value=65535 stack");
}



for (x = 0; x < n; x++) {
	print("x = " + x);
	offsetX = x * width / n;
	title_crop = "Img_" + x;
	//print(offsetX);
	selectImage(id);

	run("Duplicate...", "duplicate");
	rename(title_crop);
	makeRectangle(offsetX, 0, tileWidth, height);
	run("Crop");

	if (x>0){
		selectImage("WhiteLine");
	 	run("Duplicate...", "duplicate");
	 	rename("InsertWhiteLine");
	 	
	 	run("Combine...", "stack1=InsertWhiteLine stack2=" + title_crop + " combine");
	 	rename("Add");

 		run("Combine...", "stack1=" + new_title + " stack2=Add combine");
	 	rename(new_title);
	}
 	rename(new_title);
	
}

selectImage("WhiteLine");
close();
