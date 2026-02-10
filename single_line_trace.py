import fitz
import cv2
import numpy as np
import os

def extract_page_as_image(pdf_path, output_path="extracted/page.png", dpi=300):
    """将PDF页面提取为高分辨率图片"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    pixmap = page.get_pixmap(matrix=mat)
    pixmap.save(output_path)
    
    doc.close()
    return output_path

def get_skeleton(img):
    """骨架化处理 - 将线条细化为中心线"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    skeleton = np.zeros(binary.shape, np.uint8)
    
    size = np.size(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    done = False
    while not done:
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()
        
        zeros = size - cv2.countNonZero(binary)
        if zeros == size:
            done = True
    
    kernel2 = np.ones((2, 2), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel2)
    
    return skeleton

def skeleton_to_svg(skeleton, output_path, width, height):
    """将骨架化结果转为SVG"""
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    total_points = sum(len(c) for c in contours)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Single Line Skeleton - {len(contours)} paths</title>
  <desc>Skeletonized lines (single line tracing)</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <g id="lines" stroke="black" fill="none" stroke-width="1">
'''
    
    for i, contour in enumerate(contours):
        points = contour.reshape(-1, 2)
        if len(points) < 2:
            continue
        
        if len(points) == 2:
            svg_content += f'    <line id="l{i}" x1="{points[0][0]}" y1="{points[0][1]}" '
            svg_content += f'x2="{points[1][0]}" y2="{points[1][1]}"/>\n'
        else:
            svg_content += f'    <path id="p{i}" d="'
            svg_content += f"M {points[0][0]} {points[0][1]} "
            for j in range(1, len(points)):
                svg_content += f"L {points[j][0]} {points[j][1]} "
            svg_content += '"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    return len(contours), total_points

def skeleton_to_ai(skeleton, output_path, width, height):
    """将骨架化结果转为AI格式"""
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Single Line Skeleton Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% SINGLE LINE SKELETON
% {len(contours)} paths
% Each path is a single line (center line)

0 setgray
1 setlinewidth

'''
    
    for i, contour in enumerate(contours):
        points = contour.reshape(-1, 2)
        if len(points) < 2:
            continue
        
        ai_content += f"\n% Path {i + 1} ({len(points)} points)\n"
        ai_content += "newpath\n"
        ai_content += f"{points[0][0]} {height - points[0][1]} moveto\n"
        
        for j in range(1, len(points)):
            x, y = points[j]
            ai_content += f"{x} {height - y} lineto\n"
        
        ai_content += "closepath\n"
        ai_content += "stroke\n"
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ai_content)
    
    return len(contours)

def trace_center_lines(img, output_prefix, width, height):
    """追踪线条中心线"""
    skeleton = get_skeleton(img)
    
    print(f"    骨架化完成...")
    
    contour_count, point_count = skeleton_to_svg(skeleton, f"{output_prefix}_skeleton.svg", width, height)
    print(f"    SVG: {contour_count} 条线, {point_count} 个点")
    
    contour_count = skeleton_to_ai(skeleton, f"{output_prefix}_skeleton.ai", width, height)
    print(f"    AI: {contour_count} 条线")
    
    return contour_count

def process_pdf_single_lines(pdf_path="a.pdf"):
    """处理PDF - 单线条描边"""
    print("=" * 60)
    print("PDF 单线条描边 (骨架化 - 一条线)")
    print("=" * 60)
    
    print("\n[1/3] 提取PDF页面...")
    image_path = extract_page_as_image(pdf_path, dpi=300)
    
    print("\n[2/3] 处理图片...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    height, width = img.shape[:2]
    print(f"    图片尺寸: {width}x{height}")
    
    base_name = "extracted/single_line"
    
    print("\n    生成单线条描边...")
    trace_center_lines(img, base_name, width, height)
    
    print("\n[3/3] 完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - single_line_skeleton.svg  (SVG - 单线条)")
    print("  - single_line_skeleton.ai   (AI - 单线条)")
    print("\n使用说明:")
    print("  1. Adobe Illustrator: 打开 single_line_skeleton.ai")
    print("  2. 每条线都是独立的中心线，不是空心轮廓")
    print("  3. 线条宽度统一，适合作为描边使用")

if __name__ == "__main__":
    process_pdf_single_lines("a.pdf")
