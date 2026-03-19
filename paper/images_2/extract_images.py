# step 1 - extract pages as images (using PyMuPDF)
import fitz  # PyMuPDF

doc = fitz.open("CarPlanner.pdf")

for i in range(len(doc)):
    page = doc[i]
    pix = page.get_pixmap(dpi=300)   # high quality render
    pix.save(f"page_{i+1}.png")


# Step 2- crop images to extract figures (using PIL)
from PIL import Image

img = Image.open("page_1.png")

# Example crop for Figure 1 (adjust coordinates per page)
crop = img.crop((1200, 300, 2400, 1500))  
crop.save("figure1.png")

# step3: extract embedded images (using PyMuPDF)

for page_index in range(len(doc)):
    page = doc[page_index]
    images = page.get_images(full=True)

    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image["ext"]

        with open(f"img_{page_index}_{img_index}.{ext}", "wb") as f:
            f.write(image_bytes)