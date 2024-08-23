import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from seaborn import color_palette

def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]

def draw_outputs(img, outputs, class_names):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]

    wh = img.shape[1::-1]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='./data/fonts/futur.ttf', size=(img.size[0] + img.size[1]) // 100)

    for i in range(nums):
        color = colors[int(classes[i])]
        xywh = boxes[i]
        xyxy = xywh_to_xyxy(xywh)

        x1y1 = ((np.array(xyxy[0:2])).astype(np.int32))
        x2y2 = ((np.array(xyxy[2:4])).astype(np.int32))

        print("Top-left coordinates:", x1y1)
        print("Bottom-right coordinates:", x2y2)

        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]

        for t in np.linspace(0, 1, thickness):
            draw.rectangle([x1y1[0] - t, x1y1[1] - t, x2y2[0] - t, x2y2[1] - t], outline=tuple(color))

        confidence = '{:.2f}%'.format(objectness[i] * 100)
        text = '{} {}'.format(class_names[int(classes[i])], confidence)
        text_size = draw.textsize(text, font=font)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0], fill=tuple(color))
        draw.text((x0, y0 - text_size[1]), text, fill='black', font=font)

    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img
