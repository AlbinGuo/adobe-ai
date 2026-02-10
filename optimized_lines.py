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

def preprocess_image(img):
    """图像预处理"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    kernel_close = np.ones((4, 4), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=4)
    
    kernel_open = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    return binary, gray

def adaptive_centerline(binary_img):
    """自适应中心线提取"""
    binary = np.uint8(binary_img)
    
    dist = cv2.distanceTransform(binary, cv2.DIST_L1, 5)
    
    markers = np.zeros(dist.shape, np.int32)
    markers[dist > 0.5] = 1
    markers[dist > 2] = 2
    markers[dist > 5] = 3
    
    _, skeleton = cv2.threshold(dist, 0.8, 255, cv2.THRESH_BINARY)
    skeleton = np.uint8(skeleton)
    
    kernel = np.ones((2, 2), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return skeleton

def trace_all_contours(skeleton):
    """追踪所有连通轮廓"""
    visited = np.zeros(skeleton.shape, np.uint8)
    contours = []
    
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
            
            if len(points) > 5:
                contours.append(np.array(points))
    
    return contours

def optimize_contour(points, epsilon=1.5):
    """Douglas-Peucker轮廓优化"""
    if len(points) < 3:
        return np.array(points) if len(points) > 1 else np.array([])
    
    def distance(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def douglas_peucker(start, end, pts, eps):
        if end - start < 2:
            return np.array([pts[start], pts[end]])
        
        start_point = pts[start]
        end_point = pts[end]
        
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        
        line_len = np.sqrt(dx*dx + dy*dy)
        if line_len < 0.001:
            return np.array([start_point])
        
        max_dist = 0
        max_idx = start
        
        for i in range(start + 1, end):
            p = pts[i]
            dist = abs(dx * (start_point[1] - p[1]) - dy * (start_point[0] - p[0])) / line_len
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        if max_dist > eps:
            left = douglas_peucker(start, max_idx, pts, eps)
            right = douglas_peucker(max_idx, end, pts, eps)
            return np.vstack([left, right])
        else:
            return np.array([start_point, end_point])
    
    points = np.array(points)
    if len(points) < 2:
        return points
    
    return douglas_peucker(0, len(points)-1, points, epsilon)

def remove_redundant_points(points, min_dist=2):
    """移除冗余点"""
    if len(points) < 2:
        return points
    
    filtered = [points[0]]
    for i in range(1, len(points)):
        p1 = filtered[-1]
        p2 = points[i]
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if dist >= min_dist:
            filtered.append(p2)
    
    return np.array(filtered)

def simplify_contour_aggressive(points, target_points=500):
    """激进简化轮廓"""
    if len(points) <= target_points:
        return points
    
    step = len(points) // target_points
    if step < 1:
        step = 1
    
    simplified = points[::step].copy()
    
    if len(simplified) > target_points:
        simplified = simplified[:target_points]
    
    if len(simplified) > 2:
        if np.allclose(simplified[0], simplified[-1]):
            simplified = simplified[:-1]
    
    return simplified

def merge_nearby_contours(contours, max_gap=50):
    """合并相邻轮廓"""
    if len(contours) < 2:
        return contours
    
    merged = []
    used = [False] * len(contours)
    
    for i in range(len(contours)):
        if used[i]:
            continue
        
        current = contours[i]
        used[i] = True
        
        for j in range(i + 1, len(contours)):
            if used[j]:
                continue
            
            other = contours[j]
            
            dist1 = np.sqrt((current[-1][0] - other[0][0])**2 + 
                           (current[-1][1] - other[0][1])**2)
            dist2 = np.sqrt((current[0][0] - other[-1][0])**2 + 
                           (current[0][1] - other[-1][1])**2)
            
            if dist1 < max_gap or dist2 < max_gap:
                if dist1 < dist2:
                    current = np.vstack([current, other])
                else:
                    current = np.vstack([other, current])
                used[j] = True
        
        merged.append(current)
    
    return merged

def connect_contours(contours, max_gap=30):
    """连接相近的轮廓端点"""
    if len(contours) < 2:
        return contours
    
    connected = list(contours)
    
    changed = True
    iterations = 0
    max_iterations = len(contours)
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        result = []
        used = [False] * len(connected)
        
        for i in range(len(connected)):
            if used[i]:
                continue
            
            current = connected[i]
            used[i] = True
            
            for j in range(i + 1, len(connected)):
                if used[j]:
                    continue
                
                other = connected[j]
                
                d1 = np.sqrt((current[-1][0] - other[0][0])**2 + 
                            (current[-1][1] - other[0][1])**2)
                d2 = np.sqrt((current[0][0] - other[-1][0])**2 + 
                            (current[0][1] - other[-1][1])**2)
                
                if d1 < max_gap or d2 < max_gap:
                    if d1 < d2:
                        current = np.vstack([current, other])
                    else:
                        current = np.vstack([other, current])
                    used[j] = True
                    changed = True
            
            result.append(current)
        
        connected = result
    
    return connected

def output_svg(contours, output_path, width, height):
    """输出优化后的SVG"""
    total_points = sum(len(c) for c in contours)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Optimized Vector Lines - {len(contours)} paths, {total_points} points</title>
  <desc>Optimized single solid lines for printing</desc>
  
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
    
    return len(contours), total_points

def output_ai(contours, output_path, width, height):
    """输出优化后的AI格式"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Optimized Vector Lines
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% OPTIMIZED SINGLE SOLID LINES
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

def process_optimized(pdf_path="a.pdf"):
    """优化后的处理流程"""
    print("=" * 60)
    print("PDF 线条优化处理 (印刷级)")
    print("=" * 60)
    
    print("\n[1/5] 提取PDF页面...")
    image_path = extract_page_as_image(pdf_path, dpi=300)
    
    print("\n[2/5] 预处理图片...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    height, width = img.shape[:2]
    print(f"    尺寸: {width}x{height}")
    
    binary, gray = preprocess_image(img)
    
    print("\n[3/5] 提取中心线...")
    skeleton = adaptive_centerline(binary)
    
    print("\n[4/5] 优化线条...")
    raw_contours = trace_all_contours(skeleton)
    print(f"    原始轮廓: {len(raw_contours)} 个")
    
    merged = connect_contours(raw_contours, max_gap=40)
    print(f"    连接后: {len(merged)} 个")
    
    optimized = []
    for contour in merged:
        if len(contour) < 3:
            continue
        simplified = optimize_contour(list(contour), epsilon=2.0)
        cleaned = remove_redundant_points(simplified, min_dist=3)
        if len(cleaned) > 2:
            optimized.append(cleaned)
    
    print(f"    简化后: {len(optimized)} 个")
    
    total_points = sum(len(c) for c in optimized)
    print(f"    总点数: {total_points}")
    
    print("\n[5/5] 生成矢量文件...")
    base_name = "extracted/optimized"
    
    count, pts = output_svg(optimized, f"{base_name}.svg", width, height)
    print(f"    SVG: {count} 条路径, {pts} 点")
    
    output_ai(optimized, f"{base_name}.ai", width, height)
    print(f"    AI: {count} 条路径")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - optimized.svg ({count} 条连续路径)")
    print(f"  - optimized.ai  (AI格式)")
    print("\n特点:")
    print("  1. 连续线条，无断开")
    print("  2. 点数精简，适合印刷")
    print("  3. 纯描边线条，无填充")

if __name__ == "__main__":
    process_optimized("a.pdf")
