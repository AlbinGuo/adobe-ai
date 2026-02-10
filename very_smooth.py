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

def get_smooth_lines(img):
    """获取平滑线条"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((6, 6), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    kernel2 = np.ones((4, 4), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2, iterations=3)
    
    edges = cv2.Canny(binary, 30, 100)
    
    kernel3 = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel3, iterations=3)
    
    return edges

def get_thin_centerline(edges):
    """获取细中心线"""
    dist = cv2.distanceTransform(edges, cv2.DIST_L1, 3)
    _, skeleton = cv2.threshold(dist, 0.5, 255, cv2.THRESH_BINARY)
    skeleton = np.uint8(skeleton)
    
    kernel = np.ones((3, 3), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return skeleton

def extract_paths(skeleton, min_length=80):
    """提取路径"""
    visited = np.zeros(skeleton.shape, np.uint8)
    height, width = skeleton.shape
    
    paths = []
    
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
                paths.append(np.array(points))
    
    return paths

def savitzky_golay_smooth(points, window=11, order=3):
    """Savitzky-Golay 平滑"""
    if len(points) < window:
        return points
    
    points = np.array(points)
    
    half = window // 2
    
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - half)
        end = min(len(points), i + half + 1)
        
        window_pts = points[start:end]
        
        x = np.arange(len(window_pts))
        try:
            z = np.polyfit(x, window_pts[:, 0], order)
            p = np.poly1d(z)
            x_new = i - start
            smoothed_x = int(p(x_new))
        except:
            smoothed_x = points[i][0]
        
        try:
            z = np.polyfit(x, window_pts[:, 1], order)
            p = np.poly1d(z)
            x_new = i - start
            smoothed_y = int(p(x_new))
        except:
            smoothed_y = points[i][1]
        
        smoothed.append((smoothed_x, smoothed_y))
    
    return np.array(smoothed)

def moving_average_smooth(points, window=9):
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

def curve_fitting_smooth(points, n_points=None):
    """曲线拟合"""
    if len(points) < 4:
        return points
    
    if n_points is None:
        n_points = len(points)
    
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, n_points)
    
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    
    try:
        from scipy.interpolate import interp1d
        fx = interp1d(t, x, kind='cubic')
        fy = interp1d(t, y, kind='cubic')
        
        x_new = fx(t_new)
        y_new = fy(t_new)
        
        return np.column_stack([x_new.astype(int), y_new.astype(int)])
    except:
        return points

def simplify_path(points, tolerance=10):
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

def process_smooth(pdf_path="a.pdf"):
    """处理平滑线条"""
    print("=" * 60)
    print("PDF 平滑线条处理 (多重平滑)")
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
    
    print("\n[3/5] 提取平滑线条...")
    edges = get_smooth_lines(img)
    
    skeleton = get_thin_centerline(edges)
    
    paths = extract_paths(skeleton, min_length=60)
    print(f"    原始路径: {len(paths)} 条")
    
    print("\n[4/5] 多重平滑处理...")
    smoothed_paths = []
    for path in paths:
        if len(path) < 15:
            continue
        
        step1 = moving_average_smooth(path, window=9)
        if len(step1) < 10:
            continue
        
        step2 = savitzky_golay_smooth(step1, window=7, order=2)
        if len(step2) < 8:
            continue
        
        try:
            step3 = curve_fitting_smooth(step2)
        except:
            step3 = step2
        
        if len(step3) < 6:
            continue
        
        final = simplify_path(step3, tolerance=8)
        if len(final) > 5:
            smoothed_paths.append(final)
    
    print(f"    平滑后: {len(smoothed_paths)} 条")
    
    total_points = sum(len(p) for p in smoothed_paths)
    print(f"    总点数: {total_points}")
    
    print("\n[5/5] 生成矢量文件...")
    base_name = "extracted/very_smooth"
    
    output_svg(smoothed_paths, f"{base_name}.svg", width, height)
    output_ai(smoothed_paths, f"{base_name}.ai", width, height)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - very_smooth.svg")
    print(f"  - very_smooth.ai")
    print(f"\n处理流程:")
    print("  1. 高斯模糊降噪")
    print("  2. 形态学闭运算连接断线")
    print("  3. 移动平均平滑")
    print("  4. Savitzky-Golay平滑")
    print("  5. 曲线拟合")
    print("  6. 路径简化")

def output_svg(paths, output_path, width, height):
    """输出SVG"""
    total_points = sum(len(p) for p in paths)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Very Smooth Lines - {len(paths)} paths, {total_points} points</title>
  <desc>Multi-smoothed clean lines for printing</desc>
  
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
%%Creator: Very Smooth Line Tracer
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% VERY SMOOTH LINES
% Multi-smoothing processed

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
    process_smooth("a.pdf")
