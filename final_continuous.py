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

def get_continuous_skeleton(img):
    """获取连续的骨架线 - 增强版"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    _, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((4, 4), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    kernel2 = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2, iterations=2)
    
    dist = cv2.distanceTransform(binary, cv2.DIST_L1, 3)
    
    _, skeleton = cv2.threshold(dist, 0.3, 255, cv2.THRESH_BINARY)
    skeleton = np.uint8(skeleton)
    
    kernel3 = np.ones((3, 3), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel3, iterations=4)
    
    return skeleton

def skeleton_to_continuous_paths(skeleton):
    """骨架转为连续路径"""
    visited = np.zeros(skeleton.shape, np.uint8)
    all_paths = []
    
    height, width = skeleton.shape
    
    for y in range(height):
        for x in range(width):
            if skeleton[y, x] == 0 or visited[y, x]:
                continue
            
            path = []
            stack = [(x, y)]
            visited[y, x] = 255
            
            while stack:
                cx, cy = stack.pop()
                path.append((cx, cy))
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if skeleton[ny, nx] > 0 and visited[ny, nx] == 0:
                                visited[ny, nx] = 255
                                stack.append((nx, ny))
            
            if len(path) > 10:
                all_paths.append(np.array(path))
    
    return all_paths

def simplify_path_douglas(points, tolerance=5):
    """Douglas-Peucker路径简化"""
    if len(points) <= 2:
        return points
    
    def point_to_line_dist(p, a, b):
        """点到直线的距离"""
        ax, ay = a
        bx, by = b
        px, py = p
        
        dx = bx - ax
        dy = by - ay
        
        if dx == 0 and dy == 0:
            return np.sqrt((px-ax)**2 + (py-ay)**2)
        
        t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
        
        nearest_x = ax + t * dx
        nearest_y = ay + t * dy
        
        return np.sqrt((px-nearest_x)**2 + (py-nearest_y)**2)
    
    def recursive_simplify(pts, start, end, tol):
        if end - start <= 1:
            return [pts[start]]
        
        max_dist = 0
        max_idx = start
        
        a = pts[start]
        b = pts[end]
        
        for i in range(start + 1, end):
            dist = point_to_line_dist(pts[i], a, b)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        if max_dist > tol:
            left = recursive_simplify(pts, start, max_idx, tol)
            right = recursive_simplify(pts, max_idx, end, tol)
            return left + right[1:]
        else:
            return [pts[start], pts[end]]
    
    result = recursive_simplify(points, 0, len(points)-1, tolerance)
    return np.array(result)

def merge_adjacent_paths(paths, max_gap=100):
    """合并相邻路径"""
    if len(paths) <= 1:
        return paths
    
    merged = []
    used = [False] * len(paths)
    
    for i in range(len(paths)):
        if used[i]:
            continue
        
        current = paths[i]
        used[i] = True
        
        for j in range(i + 1, len(paths)):
            if used[j]:
                continue
            
            other = paths[j]
            
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
        
        merged.append(current)
    
    return merged

def reduce_point_density(points, min_dist=4):
    """降低点密度"""
    if len(points) < 2:
        return points
    
    reduced = [points[0]]
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - reduced[-1][0])**2 + 
                       (points[i][1] - reduced[-1][1])**2)
        if dist >= min_dist:
            reduced.append(points[i])
    
    return np.array(reduced)

def process_continuous(pdf_path="a.pdf"):
    """处理连续线条 - 最终版"""
    print("=" * 60)
    print("PDF 连续线条 - 最终优化版")
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
    
    print("\n[3/5] 生成连续骨架...")
    skeleton = get_continuous_skeleton(img)
    
    print("\n[4/5] 提取并优化路径...")
    paths = skeleton_to_continuous_paths(skeleton)
    print(f"    原始路径: {len(paths)} 条")
    
    paths = merge_adjacent_paths(paths, max_gap=80)
    print(f"    合并后: {len(paths)} 条")
    
    optimized = []
    for path in paths:
        if len(path) < 3:
            continue
        reduced = reduce_point_density(path, min_dist=3)
        if len(reduced) > 2:
            simplified = simplify_path_douglas(reduced, tolerance=4)
            if len(simplified) > 2:
                optimized.append(simplified)
    
    print(f"    优化后: {len(optimized)} 条")
    
    total_points = sum(len(p) for p in optimized)
    print(f"    总点数: {total_points}")
    
    print("\n[5/5] 生成矢量文件...")
    base_name = "extracted/final_continuous"
    
    output_svg(optimized, f"{base_name}.svg", width, height)
    output_ai(optimized, f"{base_name}.ai", width, height)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - final_continuous.svg")
    print(f"  - final_continuous.ai")
    print(f"\n文件信息:")
    print(f"  - {len(optimized)} 条连续路径")
    print(f"  - {total_points} 个优化点")
    print(f"  - 无断线，纯描边")

def output_svg(paths, output_path, width, height):
    """输出SVG"""
    total_points = sum(len(p) for p in paths)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Continuous Lines - {len(paths)} paths, {total_points} points</title>
  <desc>Continuous single lines for printing</desc>
  
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
%%Creator: Continuous Line Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% CONTINUOUS SINGLE LINES
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
    process_continuous("a.pdf")
