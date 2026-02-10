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

def get_edges(img):
    """获取边缘"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 35, 100, apertureSize=3)
    
    kernel = np.ones((4, 4), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return edges

def edges_to_paths_fast(edges):
    """快速边缘转路径"""
    visited = np.zeros(edges.shape, np.uint8)
    paths = []
    h, w = edges.shape
    
    for y in range(h):
        for x in range(w):
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
                        if 0 <= nx < w and 0 <= ny < h:
                            if edges[ny, nx] > 0 and visited[ny, nx] == 0:
                                visited[ny, nx] = 255
                                stack.append((nx, ny))
            
            if len(points) > 30:
                paths.append(np.array(points))
    
    return paths

def simple_simplify(points, tolerance=20):
    """简单简化"""
    if len(points) < 3:
        return points
    
    simplified = [points[0]]
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - simplified[-1][0])**2 + 
                       (points[i][1] - simplified[-1][1])**2)
        if dist >= tolerance:
            simplified.append(points[i])
    
    return np.array(simplified)

def simple_smooth(points, window=11):
    """简单移动平均"""
    if len(points) < window:
        return points
    
    smoothed = []
    half = window // 2
    
    for i in range(len(points)):
        start = max(0, i - half)
        end = min(len(points), i + half)
        
        window_pts = points[start:end]
        
        avg_x = int(round(np.mean([p[0] for p in window_pts])))
        avg_y = int(round(np.mean([p[1] for p in window_pts])))
        
        smoothed.append((avg_x, avg_y))
    
    return np.array(smoothed)

def process_fast(pdf_path="a.pdf", line_width=4):
    """快速处理"""
    print("=" * 60)
    print("PDF 平滑线条处理")
    print(f"线条宽度: {line_width}px")
    print("=" * 60)
    
    print("\n[1/4] 提取页面...")
    image_path = extract_page_as_image(pdf_path, dpi=150)
    
    print("\n[2/4] 处理图片...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取")
        return
    
    h, w = img.shape[:2]
    print(f"    尺寸: {w}x{h}")
    
    edges = get_edges(img)
    paths = edges_to_paths_fast(edges)
    print(f"    路径: {len(paths)} 条")
    
    print("\n[3/4] 简化...")
    final = []
    for path in paths:
        if len(path) < 10:
            continue
        
        s1 = simple_smooth(path, window=9)
        if len(s1) < 8:
            continue
        
        s2 = simple_simplify(s1, tolerance=15)
        if len(s2) > 3:
            final.append(s2)
    
    print(f"    优化后: {len(final)} 条")
    total = sum(len(p) for p in final)
    print(f"    总点数: {total}")
    
    print("\n[4/4] 生成文件...")
    base = f"extracted/result_{line_width}px"
    
    write_svg(final, f"{base}.svg", w, h, line_width)
    write_ai(final, f"{base}.ai", w, h, line_width)
    
    print("\n完成!")
    print(f"  - {base}.svg")
    print(f"  - {base}.ai")

def write_svg(path_list, output_path, w, h, lw):
    pts = sum(len(p) for p in path_list)
    
    s = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <title>Smooth Lines - {len(path_list)} paths, width: {lw}px</title>
  <rect width="100%" height="100%" fill="white"/>
  <g stroke="black" fill="none" stroke-width="{lw}" stroke-linecap="round" stroke-linejoin="round">
'''
    
    for i, path_data in enumerate(path_list):
        if len(path_data) < 2:
            continue
        p = path_data.reshape(-1, 2)
        s += f'    <path id="p{i}" d="'
        s += f"M {p[0][0]} {p[0][1]} "
        for j in range(1, len(p)):
            s += f"L {p[j][0]} {p[j][1]} "
        s += '"/>\n'
    
    s += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(s)
    
    print(f"    SVG: {len(path_list)} 条, {pts} 点")

def write_ai(path_list, output_path, w, h, lw):
    a = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Smooth Lines
%%Pages: 1
%%BoundingBox: 0 0 {w} {h}
<< /PageSize [{w} {h}] >> setpagedevice
0 setgray
{lw / 72:.3f} setlinewidth
1 setlinecap
1 setlinejoin

'''
    hi = int(h)
    
    for i, path_data in enumerate(path_list):
        if len(path_data) < 2:
            continue
        p = path_data.reshape(-1, 2)
        a += f"\n% Path {i + 1} ({len(p)} pts)\n"
        a += "newpath\n"
        a += f"{p[0][0]} {hi - p[0][1]} moveto\n"
        for j in range(1, len(p)):
            x, y = p[j]
            a += f"{x} {hi - y} lineto\n"
        a += "stroke\n"
    
    a += "\nshowpage\n%%EndDocument\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(a)
    
    print(f"    AI: {len(path_list)} 条")

if __name__ == "__main__":
    process_fast("a.pdf", line_width=4)
