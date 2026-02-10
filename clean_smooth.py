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

def get_clean_lines(img):
    """获取干净的线条 - 去除毛刺"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    edges = cv2.Canny(blurred, 40, 120, apertureSize=3)
    
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=2)
    
    kernel2 = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel2, iterations=2)
    
    return edges

def get_thin_lines(edges):
    """细化线条为单像素"""
    edges = np.uint8(edges)
    
    dist = cv2.distanceTransform(edges, cv2.DIST_L1, 3)
    
    _, skeleton = cv2.threshold(dist, 0.5, 255, cv2.THRESH_BINARY)
    skeleton = np.uint8(skeleton)
    
    return skeleton

def remove_noise_from_skeleton(skeleton, min_length=100):
    """从骨架中移除噪点"""
    visited = np.zeros(skeleton.shape, np.uint8)
    height, width = skeleton.shape
    
    valid_segments = []
    
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
            
            if len(points) >= min_length:
                valid_segments.append(np.array(points))
    
    return valid_segments

def smooth_path(points, window=5):
    """移动平均平滑"""
    if len(points) < window:
        return points
    
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window // 2)
        end = min(len(points), i + window // 2 + 1)
        
        window_points = points[start:end]
        
        if len(window_points) > 0:
            avg_x = int(np.mean([p[0] for p in window_points]))
            avg_y = int(np.mean([p[1] for p in window_points]))
            smoothed.append((avg_x, avg_y))
    
    return np.array(smoothed)

def chaikin_smooth(points, iterations=2):
    """Chaikin's corner cutting smoothing"""
    if len(points) < 4 or iterations <= 0:
        return points
    
    def chaikin_step(pts):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_pts.append(q)
            new_pts.append(r)
        new_pts.append(pts[-1])
        return np.array(new_pts)
    
    result = points.copy()
    for _ in range(iterations):
        result = chaikin_step(result)
    
    return result

def simplify_path_points(points, tolerance=8):
    """简化路径点"""
    if len(points) < 3:
        return points
    
    simplified = [points[0]]
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - simplified[-1][0])**2 + 
                       (points[i][1] - simplified[-1][1])**2)
        if dist >= tolerance:
            simplified.append(points[i])
    
    return np.array(simplified)

def merge_close_points(points, threshold=10):
    """合并相近的点"""
    if len(points) < 2:
        return points
    
    merged = [points[0]]
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - merged[-1][0])**2 + 
                       (points[i][1] - merged[-1][1])**2)
        if dist >= threshold:
            merged.append(points[i])
    
    return np.array(merged)

def smooth_and_clean(img):
    """完整的平滑去毛刺流程"""
    edges = get_clean_lines(img)
    
    skeleton = get_thin_lines(edges)
    
    segments = remove_noise_from_skeleton(skeleton, min_length=80)
    
    smoothed_segments = []
    for segment in segments:
        if len(segment) < 10:
            continue
        
        smoothed = smooth_path(segment, window=7)
        if len(smoothed) > 4:
            chaikin = chaikin_smooth(smoothed, iterations=1)
            if len(chaikin) > 4:
                simplified = simplify_path_points(chaikin, tolerance=10)
                if len(simplified) > 4:
                    merged = merge_close_points(simplified, threshold=8)
                    if len(merged) > 4:
                        smoothed_segments.append(merged)
    
    return smoothed_segments

def output_svg(paths, output_path, width, height):
    """输出SVG"""
    total_points = sum(len(p) for p in paths)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Clean Lines - {len(paths)} paths, {total_points} points</title>
  <desc>Clean smooth lines without noise for printing</desc>
  
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
    
    return total_points

def output_ai(paths, output_path, width, height):
    """输出AI格式"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Clean Line Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% CLEAN SMOOTH LINES
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

def process_clean_lines(pdf_path="a.pdf"):
    """处理干净的线条"""
    print("=" * 60)
    print("PDF 干净线条处理 (无毛刺)")
    print("=" * 60)
    
    print("\n[1/4] 提取PDF页面...")
    image_path = extract_page_as_image(pdf_path, dpi=300)
    
    print("\n[2/4] 读取图片...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    height, width = img.shape[:2]
    print(f"    尺寸: {width}x{height}")
    
    print("\n[3/4] 平滑去毛刺...")
    paths = smooth_and_clean(img)
    print(f"    提取路径: {len(paths)} 条")
    
    total_points = sum(len(p) for p in paths)
    print(f"    总点数: {total_points}")
    
    print("\n[4/4] 生成矢量文件...")
    base_name = "extracted/clean_smooth"
    
    pts = output_svg(paths, f"{base_name}.svg", width, height)
    print(f"    SVG: {len(paths)} 条, {pts} 点")
    
    output_ai(paths, f"{base_name}.ai", width, height)
    print(f"    AI: {len(paths)} 条")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - clean_smooth.svg")
    print(f"  - clean_smooth.ai")
    print(f"\n特点:")
    print("  1. 无毛刺噪声")
    print("  2. 平滑曲线")
    print("  3. 适合印刷")

if __name__ == "__main__":
    process_clean_lines("a.pdf")
