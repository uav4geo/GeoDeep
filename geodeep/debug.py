
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
            text = f"{score:.2f}"  # Score text
            draw.text((x1, y1 - 10), text, fill=color)  # Text above the bounding box

    img.save(output_path)