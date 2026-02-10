import fitz
import cv2
import numpy as np
import os
from scipy.interpolate import splprep, splev

def extract_page_as_image(pdf_path, output_path="extracted/page.png", dpi=300):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=mat)
    pixmap.save(output_path)
    doc.close()
    return output_path

def get_smooth_edges(img):
    """获取平滑的边缘"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    
    kernel = np.ones((6, 6), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    kernel2 = np.ones((4, 4), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel2, iterations=2)
    
    edges = cv2.dilate(edges, kernel2, iterations=2)
    
    return edges

def get_centerline(edges):
    """获取中心线"""
    edges = np.uint8(edges)
    
    dist = cv2.distanceTransform(edges, cv2.DIST_L2, 5)
    
    _, skeleton = cv2.threshold(dist, 0.3, 255, cv2.THRESH_BINARY)
    skeleton = np.uint8(skeleton)
    
    kernel = np.ones((4, 4), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return skeleton

def extract_long_paths(skeleton, min_length=150):
    """提取长路径"""
    visited = np.zeros(skeleton.shape, np.uint8)
    height, width = skeleton.shape
    
    paths = []
    
    for y in range(height):
        for x in range(width):
            if skeleton[y, x] == 0 or visited[y, x]:
                continue
            
            points = []
            queue = [(x, y)]
            visited[y, x] = 255
            
            idx = 0
            while idx < len(queue):
                cx, cy = queue[idx]
                idx += 1
                points.append((cx, cy))
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if skeleton[ny, nx] > 0 and visited[ny, nx] == 0:
                                visited[ny, nx] = 255
                                queue.append((nx, ny))
            
            if len(points) >= min_length:
                paths.append(np.array(points))
    
    return paths

def fit_smooth_curve(points, smoothing=100):
    """B样条平滑曲线"""
    if len(points) < 4:
        return points
    
    points = np.array(points)
    
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    
    try:
        tck, u = splprep([x, y], s=smoothing, k=3)
        
        u_fine = np.linspace(0, 1, len(points))
        x_smooth, y_smooth = splev(u_fine, tck)
        
        result = np.column_stack([x_smooth.astype(int), y_smooth.astype(int)])
        
        return result
    except:
        return points

def remove_noise_segments(paths, min_length=100):
    """移除噪点片段"""
    cleaned = []
    for path in paths:
        if len(path) >= min_length:
            cleaned.append(path)
    return cleaned

def simplify_with_epsilon(points, epsilon=5):
    """基于距离简化"""
    if len(points) < 3:
        return points
    
    simplified = [points[0]]
    
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - simplified[-1][0])**2 + 
                       (points[i][1] - simplified[-1][1])**2)
        if dist >= epsilon:
            simplified.append(points[i])
    
    return np.array(simplified)

def process_smooth_curves(pdf_path="a.pdf"):
    """处理平滑曲线"""
    print("=" * 60)
    print("PDF 平滑曲线处理 (B样条平滑)")
    print("=" * 60)
    
    print("\n[1/5] 提取PDF页面...")
    image_path = extract_page_as_image(pdf_path, dpi=300)
    
    print("\n[2/5] 读取图片...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    height, width = img.shape[:2]
    print(f"    尺寸: {width}x{height}")
    
    print("\n[3/5] 提取平滑边缘...")
    edges = get_smooth_edges(img)
    
    print("    获取中心线...")
    skeleton = get_centerline(edges)
    
    print("    提取路径...")
    raw_paths = extract_long_paths(skeleton, min_length=120)
    print(f"    原始路径: {len(raw_paths)} 条")
    
    print("\n[4/5] B样条平滑...")
    smoothed_paths = []
    for path in raw_paths:
        if len(path) < 10:
            continue
        
        smoothed = fit_smooth_curve(path, smoothing=len(path) * 2)
        if len(smoothed) > 5:
            simplified = simplify_with_epsilon(smoothed, epsilon=8)
            if len(simplified) > 5:
                smoothed_paths.append(simplified)
    
    print(f"    平滑后: {len(smoothed_paths)} 条")
    
    total_points = sum(len(p) for p in smoothed_paths)
    print(f"    总点数: {total_points}")
    
    print("\n[5/5] 生成矢量文件...")
    base_name = "extracted/smooth_bspline"
    
    output_svg(smoothed_paths, f"{base_name}.svg", width, height)
    output_ai(smoothed_paths, f"{base_name}.ai", width, height)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - smooth_bspline.svg")
    print(f"  - smooth_bspline.ai")
    print(f"\n特点:")
    print("  1. B样条平滑曲线")
    print("  2. 无毛刺噪声")
    print("  3. 适合印刷")

def output_svg(paths, output_path, width, height):
    """输出SVG"""
    total_points = sum(len(p) for p in paths)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Smooth Curves - {len(paths)} paths, {total_points} points</title>
  <desc>B-spline smoothed curves for printing</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <g id="paths" stroke="black" fill="none" stroke-width="0.5">
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

def output_ai(paths, output_path, width, height):
    """输出AI格式"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Smooth Curve Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% SMOOTH CURVES
% For printing - {len(paths)} paths

0 setgray
0.25 setlinewidth

'''
    
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
    process_smooth_curves("a.pdf")
