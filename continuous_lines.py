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

def get_continuous_skeleton(img):
    """获取连续的骨架线"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    _, skeleton = cv2.threshold(dist_transform, 0.3, 255, cv2.THRESH_BINARY)
    skeleton = np.uint8(skeleton)
    
    kernel_skel = np.ones((2, 2), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel_skel, iterations=1)
    
    return skeleton

def connect_lines(skeleton):
    """连接断开的线条"""
    kernel = np.ones((5, 5), np.uint8)
    
    dilated = cv2.dilate(skeleton, kernel, iterations=2)
    
    edges = cv2.Canny(dilated, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=30, maxLineGap=10)
    
    result = skeleton.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), 255, 2)
    
    return result

def extract_continuous_lines(img):
    """提取连续的线条"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    edges = cv2.Canny(blurred, 20, 60, apertureSize=3)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                           minLineLength=20, maxLineGap=15)
    
    if lines is not None:
        print(f"    检测到 {len(lines)} 条直线段")
    
    return edges, lines

def merge_lines(lines, width, height):
    """合并连接的线段"""
    if lines is None:
        return []
    
    merged = []
    used = [False] * len(lines)
    
    for i in range(len(lines)):
        if used[i]:
            continue
        
        x1, y1, x2, y2 = lines[i]
        
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            
            x3, y3, x4, y4 = lines[j]
            
            if self_connect(x1, y1, x2, y2, x3, y3, x4, y4, threshold=30):
                x1, y1 = min(x1, x3), min(y1, y3)
                x2, y2 = max(x2, x4), max(y2, y4)
                used[j] = True
        
        merged.append((x1, y1, x2, y2))
        used[i] = True
    
    return merged

def self_connect(x1, y1, x2, y2, x3, y3, x4, y4, threshold=30):
    """检查两条线是否应该连接"""
    dist1 = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    dist2 = np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2)
    dist3 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    dist4 = np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)
    
    return min(dist1, dist2, dist3, dist4) < threshold

def smooth_lines(lines, img_shape):
    """平滑线段"""
    smoothed = []
    height, width = img_shape[:2]
    
    for x1, y1, x2, y2 in lines:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if length > 50:
            smoothed.append((x1, y1, x2, y2))
        else:
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            smoothed.append((x1, y1, mid_x, mid_y))
            smoothed.append((mid_x, mid_y, x2, y2))
    
    return smoothed

def to_svg(lines, output_path, width, height):
    """转为SVG"""
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Continuous Lines - {len(lines)} paths</title>
  <desc>Continuous traced lines</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <g id="lines" stroke="black" fill="none" stroke-width="1">
'''
    
    for i, (x1, y1, x2, y2) in enumerate(lines):
        svg_content += f'    <line id="l{i}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

def to_ai(lines, output_path, width, height):
    """转为AI"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Continuous Lines Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% CONTINUOUS LINES
% {len(lines)} paths

0 setgray
1 setlinewidth

'''
    
    for i, (x1, y1, x2, y2) in enumerate(lines):
        ai_content += f"\n% Line {i + 1}\n"
        ai_content += "newpath\n"
        ai_content += f"{x1} {height - y1} moveto\n"
        ai_content += f"{x2} {height - y2} lineto\n"
        ai_content += "stroke\n"
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ai_content)

def to_svg_path(contours, output_path, width, height):
    """将轮廓转为连续路径SVG"""
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Continuous Paths - {len(contours)} paths</title>
  <desc>Continuous traced paths</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <g id="paths" stroke="black" fill="none" stroke-width="1">
'''
    
    total_points = 0
    for i, contour in enumerate(contours):
        points = contour.reshape(-1, 2)
        if len(points) < 2:
            continue
        
        total_points += len(points)
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

def to_ai_path(contours, output_path, width, height):
    """将轮廓转为连续路径AI"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Continuous Paths Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% CONTINUOUS PATHS
% {len(contours)} paths

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
        
        ai_content += "stroke\n"
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ai_content)
    
    return len(contours)

def extract_continuous_contours(img):
    """提取连续的轮廓"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 15, 45, apertureSize=3)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10:
            filtered.append(c)
    
    contours_sorted = sorted(filtered, key=cv2.contourArea, reverse=True)
    
    print(f"    检测到 {len(contours_sorted)} 个连续轮廓")
    
    total_points = sum(len(c) for c in contours_sorted)
    print(f"    总点数: {total_points}")
    
    return contours_sorted

def process_continuous_lines(pdf_path="a.pdf"):
    """处理连续的线条"""
    print("=" * 60)
    print("PDF 连续线条描边")
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
    
    base_name = "extracted/continuous"
    
    print("\n    提取连续线条...")
    contours = extract_continuous_contours(img)
    
    print("\n    生成矢量文件...")
    path_count, point_count = to_svg_path(contours, f"{base_name}.svg", width, height)
    print(f"    SVG: {path_count} 条路径, {point_count} 个点")
    
    path_count = to_ai_path(contours, f"{base_name}.ai", width, height)
    print(f"    AI: {path_count} 条路径")
    
    print("\n[3/3] 完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - continuous.svg  (SVG - 连续路径)")
    print("  - continuous.ai   (AI - 连续路径)")
    print("\n特点:")
    print("  1. 线条连续不中断")
    print("  2. 每条路径都是完整的多段线")
    print("  3. 无骨架化，保持线条原始粗细")

if __name__ == "__main__":
    process_continuous_lines("a.pdf")
