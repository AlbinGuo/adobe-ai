import fitz
import cv2
import numpy as np
import os

def extract_page_as_image(pdf_path, output_path="extracted/page.png", dpi=200):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=mat)
    pixmap.save(output_path)
    doc.close()
    return output_path

def extract_and_smooth_lines(img, line_width=4):
    """提取并平滑线条"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    
    kernel_size = line_width // 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    smoothed = cv2.GaussianBlur(dilated, (5, 5), 1)
    
    kernel2 = np.ones((3, 3), np.uint8)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel2, iterations=2)
    
    return smoothed

def trace_skeleton(skeleton):
    """追踪骨架"""
    visited = np.zeros(skeleton.shape, np.uint8)
    paths = []
    height, width = skeleton.shape
    
    for y in range(height):
        for x in range(width):
            if skeleton[y, x] == 0 or visited[y, x]:
                continue
            
            points = []
            stack = [(x, y)]
            visited[y, x] = 255
            
            while stack:
                cx, cy = stack.pop()
                points.append((cx, cy))
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if skeleton[ny, nx] > 0 and visited[ny, nx] == 0:
                                visited[ny, nx] = 255
                                stack.append((nx, ny))
            
            if len(points) > 50:
                paths.append(np.array(points))
    
    return paths

def smooth_path(points, window=7):
    """移动平均平滑"""
    if len(points) < window:
        return points
    
    smoothed = []
    half = window // 2
    
    for i in range(len(points)):
        start = max(0, i - half)
        end = min(len(points), i + half + 1)
        
        window_pts = points[start:end]
        
        avg_x = int(np.mean([p[0] for p in window_pts]))
        avg_y = int(np.mean([p[1] for p in window_pts]))
        
        smoothed.append((avg_x, avg_y))
    
    return np.array(smoothed)

def simplify_path(points, tolerance=5):
    """简化路径"""
    if len(points) < 3:
        return points
    
    simplified = [points[0]]
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - simplified[-1][0])**2 + 
                       (points[i][1] - simplified[-1][1])**2)
        if dist >= tolerance:
            simplified.append(points[i])
    
    return np.array(simplified)

def process_lines(pdf_path="a.pdf", line_width=4):
    """处理线条"""
    print("=" * 60)
    print("PDF 平滑线条处理")
    print(f"目标线条宽度: {line_width}px")
    print("=" * 60)
    
    print("\n[1/5] 提取PDF页面...")
    image_path = extract_page_as_image(pdf_path, dpi=200)
    
    print("\n[2/5] 读取图片...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    height, width = img.shape[:2]
    print(f"    尺寸: {width}x{height}")
    
    print("\n[3/5] 提取并平滑线条...")
    smoothed = extract_and_smooth_lines(img, line_width)
    
    print("    追踪骨架...")
    paths = trace_skeleton(smoothed)
    print(f"    原始路径: {len(paths)} 条")
    
    print("\n[4/5] 优化路径...")
    optimized = []
    for path in paths:
        if len(path) < 20:
            continue
        
        smoothed_path = smooth_path(path, window=7)
        if len(smoothed_path) < 15:
            continue
        
        simplified = simplify_path(smoothed_path, tolerance=4)
        if len(simplified) > 5:
            optimized.append(simplified)
    
    print(f"    优化后: {len(optimized)} 条")
    
    total_points = sum(len(p) for p in optimized)
    print(f"    总点数: {total_points}")
    
    print("\n[5/5] 生成矢量文件...")
    base_name = f"extracted/smooth_{line_width}px"
    
    output_svg(optimized, f"{base_name}.svg", width, height, line_width)
    output_ai(optimized, f"{base_name}.ai", width, height, line_width)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {base_name}.svg")
    print(f"  - {base_name}.ai")
    print(f"\n特点:")
    print(f"  1. 目标宽度: {line_width}px")
    print(f"  2. 平滑曲线，无毛刺")
    print(f"  3. 连续线条，无断开")

def output_svg(paths, output_path, width, height, line_width):
    """输出SVG"""
    total_points = sum(len(p) for p in paths)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Smooth Lines - {len(paths)} paths, width: {line_width}px</title>
  <desc>Smooth continuous lines for printing</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <g id="lines" stroke="black" fill="none" stroke-width="{line_width}" stroke-linecap="round" stroke-linejoin="round">
'''
    
    for i, path in enumerate(paths):
        if len(path) < 2:
            continue
        
        points = path.reshape(-1, 2)
        
        svg_content += f'    <path id="p{i}" d="'
        svg_content += f"M {points[0][0]} {points[0][1]} "
        for j in range(1, len(points)):
            svg_content += f"L {points[j][0]} {points[j][1]} "
        svg_content += '"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    print(f"    SVG: {len(paths)} 条, {total_points} 点")

def output_ai(paths, output_path, width, height, line_width):
    """输出AI格式"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Smooth Lines Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% SMOOTH CONTINUOUS LINES
% {len(paths)} paths
% Line width: {line_width}px

0 setgray
{line_width / 72} setlinewidth
1 setlinecap
1 setlinejoin

'''
    height = int(height)
    
    for i, path in enumerate(paths):
        if len(path) < 2:
            continue
        
        points = path.reshape(-1, 2)
        
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
    
    print(f"    AI: {len(paths)} 条")

if __name__ == "__main__":
    process_lines("a.pdf", line_width=4)
