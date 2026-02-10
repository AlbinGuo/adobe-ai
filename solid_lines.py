import fitz
import cv2
import numpy as np
import os

def extract_page_as_image(pdf_path, output_path="extracted/page.png", dpi=300):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=mat)
    pixmap.save(output_path)
    doc.close()
    return output_path

def adaptive_thinning(binary):
    """自适应骨架化"""
    binary = np.uint8(binary)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    dist = cv2.distanceTransform(binary, cv2.DIST_L1, 3)
    
    local_max = np.zeros(dist.shape, np.uint8)
    local_max[dist > 0] = 255
    
    _, markers = cv2.connectedComponents(local_max)
    
    return local_max, dist

def trace_center_lines(img):
    """追踪线条中心线"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    _, binary = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    _, skeleton = cv2.threshold(binary, 127, 1, cv2.THRESH_BINARY)
    
    kernel2 = np.ones((2, 2), np.uint8)
    skeleton = cv2.erode(skeleton, kernel2, iterations=1)
    skeleton = np.uint8(skeleton) * 255
    
    return skeleton, binary

def lines_to_vector_lines(skeleton, width, height):
    """骨架转为矢量线段"""
    lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, threshold=25,
                            minLineLength=15, maxLineGap=8)
    
    if lines is None:
        return []
    
    vector_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length > 5:
            vector_lines.append((x1, y1, x2, y2))
    
    return vector_lines

def lines_to_contours(skeleton, width, height):
    """骨架转为连续轮廓"""
    visited = np.zeros(skeleton.shape, np.uint8)
    contours = []
    
    height_s, width_s = skeleton.shape
    
    for y in range(height_s):
        for x in range(width_s):
            if skeleton[y, x] == 0 or visited[y, x]:
                continue
            
            points = [(x, y)]
            visited[y, x] = 255
            
            queue = [(x, y)]
            idx = 0
            
            while idx < len(queue):
                cx, cy = queue[idx]
                idx += 1
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width_s and 0 <= ny < height_s:
                            if skeleton[ny, nx] > 0 and visited[ny, nx] == 0:
                                visited[ny, nx] = 255
                                points.append((nx, ny))
                                queue.append((nx, ny))
            
            if len(points) > 3:
                contours.append(np.array(points))
    
    return contours

def simplify_contour(points, tolerance=2):
    """简化轮廓点"""
    if len(points) <= 2:
        return points
    
    simplified = [points[0]]
    
    for i in range(1, len(points)):
        last = simplified[-1]
        curr = points[i]
        dist = np.sqrt((curr[0] - last[0])**2 + (curr[1] - last[1])**2)
        if dist >= tolerance:
            simplified.append(curr)
    
    if len(simplified) > 2 and np.sqrt((simplified[-1][0] - simplified[0][0])**2 + 
                                        (simplified[-1][1] - simplified[0][1])**2) < tolerance:
        simplified.pop()
    
    return simplified

def output_svg_lines(lines, output_path, width, height):
    """输出SVG线段"""
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Solid Lines - {len(lines)} lines</title>
  <desc>Single solid lines for printing</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <g id="lines" stroke="black" fill="none" stroke-width="0.5">
'''
    
    for i, (x1, y1, x2, y2) in enumerate(lines):
        svg_content += f'    <line id="l{i}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

def output_ai_lines(lines, output_path, width, height):
    """输出AI线段"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Solid Line Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% SOLID SINGLE LINES
% For printing - {len(lines)} lines

0 setgray
0.25 setlinewidth

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

def output_svg_contours(contours, output_path, width, height):
    """输出SVG轮廓"""
    total_points = 0
    for contour in contours:
        total_points += len(contour)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Solid Contours - {len(contours)} paths, {total_points} points</title>
  <desc>Single solid lines for printing</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <g id="paths" stroke="black" fill="none" stroke-width="0.5">
'''
    
    for i, contour in enumerate(contours):
        if len(contour) < 2:
            continue
        
        points = contour.reshape(-1, 2)
        
        svg_content += f'    <path id="p{i}" d="'
        svg_content += f"M {points[0][0]} {points[0][1]} "
        for j in range(1, len(points)):
            svg_content += f"L {points[j][0]} {points[j][1]} "
        svg_content += '"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    return total_points

def output_ai_contours(contours, output_path, width, height):
    """输出AI轮廓"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Solid Contour Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% SOLID SINGLE LINES
% For printing - {len(contours)} paths

0 setgray
0.25 setlinewidth

'''
    
    for i, contour in enumerate(contours):
        if len(contour) < 2:
            continue
        
        points = contour.reshape(-1, 2)
        
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

def process_for_printing(pdf_path="a.pdf"):
    """处理PDF线条用于印刷"""
    print("=" * 60)
    print("PDF 实体单线条处理 (用于印刷)")
    print("=" * 60)
    
    print("\n[1/4] 提取PDF页面...")
    image_path = extract_page_as_image(pdf_path, dpi=300)
    
    print("\n[2/4] 读取图片...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    height, width = img.shape[:2]
    print(f"    图片尺寸: {width}x{height}")
    
    print("\n[3/4] 骨架化线条...")
    skeleton, binary = trace_center_lines(img)
    
    print("\n    提取矢量...")
    vector_lines = lines_to_vector_lines(skeleton, width, height)
    print(f"    霍夫变换: {len(vector_lines)} 条线段")
    
    contours = lines_to_contours(skeleton, width, height)
    print(f"    轮廓追踪: {len(contours)} 条连续路径")
    
    total_points = sum(len(c) for c in contours)
    print(f"    总点数: {total_points}")
    
    print("\n[4/4] 生成矢量文件...")
    base_name = "extracted/solid"
    
    output_svg_lines(vector_lines, f"{base_name}_lines.svg", width, height)
    print(f"    SVG线段: {len(vector_lines)} 条")
    
    output_ai_lines(vector_lines, f"{base_name}_lines.ai", width, height)
    print(f"    AI线段: {len(vector_lines)} 条")
    
    point_count = output_svg_contours(contours, f"{base_name}_contours.svg", width, height)
    print(f"    SVG轮廓: {len(contours)} 条, {point_count} 点")
    
    output_ai_contours(contours, f"{base_name}_contours.ai", width, height)
    print(f"    AI轮廓: {len(contours)} 条")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - solid_lines.svg  ({len(vector_lines)} 条独立线段)")
    print(f"  - solid_lines.ai   (AI格式)")
    print(f"  - solid_contours.svg ({len(contours)} 条连续路径)")
    print(f"  - solid_contours.ai  (AI格式)")
    print("\n使用建议:")
    print("  1. 优先使用 solid_contours.ai (连续路径)")
    print("  2. 线条为黑色描边，无填充")
    print("  3. 适合直接发送印刷厂")

if __name__ == "__main__":
    process_for_printing("a.pdf")
