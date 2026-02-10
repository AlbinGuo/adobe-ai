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
    """获取连续的骨架线"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=6)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    dist = cv2.distanceTransform(binary, cv2.DIST_L1, 5)
    
    _, skeleton = cv2.threshold(dist, 0.5, 255, cv2.THRESH_BINARY)
    skeleton = np.uint8(skeleton)
    
    kernel2 = np.ones((3, 3), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel2, iterations=3)
    
    return skeleton

def extract_endpoints(skeleton):
    """提取骨架端点"""
    height, width = skeleton.shape
    
    kernel = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=np.uint8)
    
    neighbors = cv2.filter2D(skeleton, -1, kernel)
    
    endpoints = np.zeros(skeleton.shape, np.uint8)
    endpoints[(skeleton > 0) & (neighbors == 255)] = 255
    
    endpoints = cv2.medianBlur(endpoints, 3)
    
    contours, _ = cv2.findContours(endpoints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    endpoint_points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            endpoint_points.append((cx, cy))
    
    return np.array(endpoint_points)

def find_close_endpoints(endpoints, max_dist=30):
    """找到需要连接的相邻端点"""
    n = len(endpoints)
    if n < 2:
        return []
    
    connections = []
    used = set()
    
    for i in range(n):
        if i in used:
            continue
        
        min_dist = max_dist
        closest = -1
        
        for j in range(i + 1, n):
            if j in used:
                continue
            
            dist = np.sqrt((endpoints[i][0] - endpoints[j][0])**2 + 
                          (endpoints[i][1] - endpoints[j][1])**2)
            if dist < min_dist:
                min_dist = dist
                closest = j
        
        if closest != -1:
            connections.append((endpoints[i], endpoints[closest]))
            used.add(i)
            used.add(closest)
    
    return connections

def connect_endpoints(skeleton, connections, height, width):
    """连接端点"""
    result = skeleton.copy()
    
    for (x1, y1), (x2, y2) in connections:
        cv2.line(result, (x1, y1), (x2, y2), 255, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return result

def skeleton_to_contours(skeleton):
    """骨架转为连续轮廓"""
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

def remove_short_segments(contours, min_length=20):
    """移除过短的线段"""
    filtered = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        total_len = 0
        for i in range(1, len(contour)):
            total_len += np.sqrt((contour[i][0] - contour[i-1][0])**2 + 
                                 (contour[i][1] - contour[i-1][1])**2)
        if total_len >= min_length:
            filtered.append(contour)
    
    return filtered

def simplify_contour(points, tolerance=3):
    """简化轮廓"""
    if len(points) < 3:
        return points
    
    simplified = [points[0]]
    for i in range(1, len(points)):
        dist = np.sqrt((points[i][0] - simplified[-1][0])**2 + 
                       (points[i][1] - simplified[-1][1])**2)
        if dist >= tolerance:
            simplified.append(points[i])
    
    return np.array(simplified)

def output_svg(contours, output_path, width, height):
    """输出SVG"""
    total_points = sum(len(c) for c in contours)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Continuous Lines - {len(contours)} paths, {total_points} points</title>
  <desc>Continuous single lines for printing</desc>
  
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

def output_ai(contours, output_path, width, height):
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

def process_continuous(pdf_path="a.pdf"):
    """处理连续线条"""
    print("=" * 60)
    print("PDF 连续线条处理 (无断线)")
    print("=" * 60)
    
    print("\n[1/6] 提取PDF页面...")
    image_path = extract_page_as_image(pdf_path, dpi=300)
    
    print("\n[2/6] 读取图片...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    height, width = img.shape[:2]
    print(f"    尺寸: {width}x{height}")
    
    print("\n[3/6] 获取骨架线...")
    skeleton = get_continuous_skeleton(img)
    
    print("\n[4/6] 检测并连接断线...")
    endpoints = extract_endpoints(skeleton)
    print(f"    检测到 {len(endpoints)} 个端点")
    
    connections = find_close_endpoints(endpoints, max_dist=40)
    print(f"    需要连接 {len(connections)} 处断线")
    
    if connections:
        skeleton = connect_endpoints(skeleton, connections, height, width)
    
    print("\n[5/6] 提取连续轮廓...")
    contours = skeleton_to_contours(skeleton)
    print(f"    原始轮廓: {len(contours)} 个")
    
    contours = remove_short_segments(contours, min_length=15)
    print(f"    过滤后: {len(contours)} 个")
    
    total_points = sum(len(c) for c in contours)
    print(f"    总点数: {total_points}")
    
    print("\n[6/6] 生成矢量文件...")
    base_name = "extracted/continuous_no_gap"
    
    pts = output_svg(contours, f"{base_name}.svg", width, height)
    print(f"    SVG: {len(contours)} 条, {pts} 点")
    
    output_ai(contours, f"{base_name}.ai", width, height)
    print(f"    AI: {len(contours)} 条")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - continuous_no_gap.svg")
    print(f"  - continuous_no_gap.ai")
    print("\n特点:")
    print("  1. 端点已自动连接，无断线")
    print("  2. 纯描边线条，无填充")
    print("  3. 适合印刷")

if __name__ == "__main__":
    process_continuous("a.pdf")
