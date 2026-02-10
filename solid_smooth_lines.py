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
    """获取平滑的线条区域"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    kernel2 = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2, iterations=2)
    
    return binary

def get_smooth_contours(binary):
    """获取平滑的轮廓"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    smoothed_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:
            continue
        
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        smoothed_contours.append(approx)
    
    return smoothed_contours

def smooth_contour_polygon(contour, iterations=2):
    """多边形平滑"""
    for _ in range(iterations):
        new_contour = []
        n = len(contour)
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            prev_pt = contour[prev_idx][0]
            curr_pt = contour[i][0]
            next_pt = contour[next_idx][0]
            
            avg_x = (prev_pt[0] + curr_pt[0] + next_pt[0]) // 3
            avg_y = (prev_pt[1] + curr_pt[1] + next_pt[1]) // 3
            
            new_contour.append([[avg_x, avg_y]])
        
        contour = np.array(new_contour)
    
    return contour

def fill_contour_svg(contour, width, height, line_width=3):
    """填充轮廓为实心线条"""
    points = contour.reshape(-1, 2)
    
    svg_path = f"M {points[0][0]} {points[0][1]}"
    for i in range(1, len(points)):
        svg_path += f" L {points[i][0]} {points[i][1]}"
    svg_path += " Z"
    
    return svg_path

def contours_to_svg(contours, output_path, width, height, line_width=3):
    """将轮廓转换为SVG"""
    total_contours = len(contours)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Smooth Solid Lines - {total_contours} shapes</title>
  <desc>Smooth filled lines for printing, width: {line_width}px</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <g id="lines" stroke="black" fill="none" stroke-width="{line_width}" stroke-linejoin="round" stroke-linecap="round">
'''
    
    for i, contour in enumerate(contours):
        contour = smooth_contour_polygon(contour)
        points = contour.reshape(-1, 2)
        
        if len(points) < 3:
            continue
        
        svg_content += f'    <path id="s{i}" d="'
        svg_content += f"M {points[0][0]} {points[0][1]} "
        for j in range(1, len(points)):
            svg_content += f"L {points[j][0]} {points[j][1]} "
        svg_content += f'Z"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

def contours_to_ai(contours, output_path, width, height, line_width=3):
    """将轮廓转换为AI格式"""
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Smooth Solid Lines
%%Title: {os.path.basename(output_path)}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% SMOOTH SOLID LINES
% {len(contours)} shapes
% Line width: {line_width}px

0 setgray
{line_width / 72} setlinewidth
1 setlinecap
1 setlinejoin

'''
    height = int(height)
    
    for i, contour in enumerate(contours):
        contour = smooth_contour_polygon(contour)
        points = contour.reshape(-1, 2)
        
        if len(points) < 3:
            continue
        
        ai_content += f"\n% Shape {i + 1} ({len(points)} points)\n"
        ai_content += "newpath\n"
        ai_content += f"{points[0][0]} {height - points[0][1]} moveto\n"
        
        for j in range(1, len(points)):
            x, y = points[j]
            ai_content += f"{x} {height - y} lineto\n"
        
        ai_content += "closepath\n"
        ai_content += "stroke\n"
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ai_content)

def process_solid_lines(pdf_path="a.pdf", line_width=4):
    """处理实心平滑线条"""
    print("=" * 60)
    print("PDF 实心平滑线条处理")
    print(f"线条宽度: {line_width}px")
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
    
    print("\n[3/5] 提取平滑线条区域...")
    binary = get_smooth_lines(img)
    
    print("\n[4/5] 获取平滑轮廓...")
    contours = get_smooth_contours(binary)
    print(f"    检测到 {len(contours)} 个线条区域")
    
    print("\n[5/5] 生成矢量文件...")
    base_name = f"extracted/solid_line_{line_width}px"
    
    contours_to_svg(contours, f"{base_name}.svg", width, height, line_width)
    contours_to_ai(contours, f"{base_name}.ai", width, height, line_width)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {base_name}.svg")
    print(f"  - {base_name}.ai")
    print(f"\n特点:")
    print(f"  1. 线条宽度: {line_width}px")
    print(f"  2. 平滑曲线，无锯齿")
    print(f"  3. 实心线条，可直接描边")
    print(f"  4. 适合印刷")

if __name__ == "__main__":
    process_solid_lines("a.pdf", line_width=4)
