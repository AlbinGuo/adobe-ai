import fitz
import cv2
import numpy as np
import os

def extract_page_as_image(pdf_path, output_path="extracted/page.png", dpi=150):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=mat)
    pixmap.save(output_path)
    doc.close()
    return output_path

def get_smooth_edges(img):
    """获取平滑边缘"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    
    edges = cv2.Canny(blurred, 40, 120, apertureSize=3)
    
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return edges

def edges_to_paths(edges):
    """边缘转路径"""
    visited = np.zeros(edges.shape, np.uint8)
    paths = []
    height, width = edges.shape
    
    for y in range(height):
        for x in range(width):
            if edges[y, x] == 0 or visited[y, x]:
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
                            if edges[ny, nx] > 0 and visited[ny, nx] == 0:
                                visited[ny, nx] = 255
                                stack.append((nx, ny))
            
            if len(points) > 30:
                paths.append(np.array(points))
    
    return paths

def ramer_douglas_pecker(points, epsilon):
    """Ramer-Douglas-Peucker 简化算法"""
    if len(points) < 3:
        return points
    
    def perpendicular_distance(point, line_start, line_end):
        if np.allclose(line_start, line_end):
            return np.linalg.norm(point - line_start)
        
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj_length = np.clip(proj_length, 0, line_len)
        nearest = line_start + proj_length * line_unitvec
        return np.linalg.norm(point - nearest)
    
    def rdp(points, start, end, eps):
        if end <= start + 1:
            return [points[start]]
        
        max_dist = 0
        max_idx = start
        
        line_start = points[start]
        line_end = points[end]
        
        for i in range(start + 1, end):
            d = perpendicular_distance(points[i], line_start, line_end)
            if d > max_dist:
                max_dist = d
                max_idx = i
        
        if max_dist > eps:
            left = rdp(points, start, max_idx, eps)
            right = rdp(points, max_idx, end, eps)
            return left[:-1] + right
        else:
            return [points[start], points[end]]
    
    return np.array(rdp(points, 0, len(points) - 1, epsilon))

def aggressive_simplify(points, tolerance=15):
    """激进简化"""
    if len(points) < 3:
        return points
    
    simplified = [points[0]]
    
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - simplified[-1][0])**2 + 
                       (points[i][1] - simplified[-1][1])**2)
        if dist >= tolerance:
            simplified.append(points[i])
    
    return np.array(simplified)

def smooth_path_moving_avg(points, window=9):
    """移动平均平滑"""
    if len(points) < window:
        return points
    
    smoothed = []
    half = window // 2
    
    for i in range(len(points)):
        start = max(0, i - half)
        end = min(len(points), i + half + 1)
        
        window_pts = points[start:end]
        
        avg_x = int(np.round(np.mean([p[0] for p in window_pts])))
        avg_y = int(np.round(np.mean([p[1] for p in window_pts])))
        
        smoothed.append((avg_x, avg_y))
    
    return np.array(smoothed)

def process_smooth(pdf_path="a.pdf", line_width=4):
    """处理平滑线条"""
    print("=" * 60)
    print("PDF 平滑线条处理 (RDP简化)")
    print(f"线条宽度: {line_width}px")
    print("=" * 60)
    
    print("\n[1/5] 提取PDF页面...")
    image_path = extract_page_as_image(pdf_path, dpi=150)
    
    print("\n[2/5] 读取图片...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    height, width = img.shape[:2]
    print(f"    尺寸: {width}x{height}")
    
    print("\n[3/5] 提取平滑边缘...")
    edges = get_smooth_edges(img)
    
    print("    提取路径...")
    paths = edges_to_paths(edges)
    print(f"    原始路径: {len(paths)} 条")
    
    print("\n[4/5] 简化和平滑...")
    final_paths = []
    for path in paths:
        if len(path) < 10:
            continue
        
        smoothed = smooth_path_moving_avg(path, window=7)
        if len(smoothed) < 8:
            continue
        
        simplified = ramer_douglas_pecker(smoothed, epsilon=12)
        if len(simplified) > 3:
            final = aggressive_simplify(simplified, tolerance=10)
            if len(final) > 3:
                final_paths.append(final)
    
    print(f"    优化后: {len(final_paths)} 条")
    
    total_points = sum(len(p) for p in final_paths)
    print(f"    总点数: {total_points}")
    
    print("\n[5/5] 生成矢量文件...")
    base_name = f"extracted/final_{line_width}px"
    
    output_svg(final_paths, f"{base_name}.svg", width, height, line_width)
    output_ai(final_paths, f"{base_name}.ai", width, height, line_width)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {base_name}.svg")
    print(f"  - {base_name}.ai")

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
    
    print(f"    SVG: {len(paths)} 条, {total_points} 点, 文件已保存")

def output_ai(paths, output_path, width, height, line_width):
    """输出AI格式"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Smooth Lines
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
{line_width / 72:.3f} setlinewidth
1 setlinecap
1 setlinejoin

'''
    h = int(height)
    
    for i, path in enumerate(paths):
        if len(path) < 2:
            continue
        
        points = path.reshape(-1, 2)
        
        ai_content += f"\n% Path {i + 1} ({len(points)} points)\n"
        ai_content += "newpath\n"
        ai_content += f"{points[0][0]} {h - points[0][1]} moveto\n"
        
        for j in range(1, len(points)):
            x, y = points[j]
            ai_content += f"{x} {h - y} lineto\n"
        
        ai_content += "stroke\n"
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ai_content)
    
    print(f"    AI: {len(paths)} 条, 文件已保存")

if __name__ == "__main__":
    process_smooth("a.pdf", line_width=4)
