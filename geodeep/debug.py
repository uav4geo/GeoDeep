import rasterio

def draw_boxes(image_path, output_path, bboxes, scores=None):
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    color = (255, 0, 0)  # Red for the bounding box

    if scores is None:
        scores = [None] * len(bboxes)

    # Draw bounding boxes with scores
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        if score is not None:
            text = f"{score:.2f}"
            draw.text((x1, y1 - 10), text, fill=color)

    img.save(output_path)

def save_raster(img, output, original_raster):
    profile = original_raster.profile
    profile.update({
        "width": img.shape[2],
        "height": img.shape[1],
        "count": img.shape[0],
    })

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(img)
        print(f"Wrote {output}")