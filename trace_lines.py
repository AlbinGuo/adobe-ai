import fitz
import cv2
import numpy as np
import os

def extract_page_as_image(pdf_path, output_path="extracted/page.png", dpi=300):
    """将PDF页面提取为高分辨率图片"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    pixmap = page.get_pixmap(matrix=mat)
    pixmap.save(output_path)
    
    doc.close()
    return output_path

def detect_black_lines(img):
    """检测黑色线条边框"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    edges = cv2.Canny(cleaned, 30, 100, apertureSize=3)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges

def contours_to_svg(points, height, contour_id):
    """将轮廓转换为SVG路径"""
    if len(points) < 2:
        return ""
    
    if len(points) == 2:
        return f'    <line id="l{contour_id}" x1="{points[0][0]}" y1="{points[0][1]}" x2="{points[1][0]}" y2="{points[1][1]}"/>\n'
    
    svg = f'    <path id="p{contour_id}" d="'
    svg += f"M {points[0][0]} {points[0][1]} "
    for j in range(1, len(points)):
        svg += f"L {points[j][0]} {points[j][1]} "
    svg += '"/>\n'
    return svg

def contours_to_ai(points, height, contour_id, area):
    """将轮廓转换为AI路径"""
    if len(points) < 2:
        return ""
    
    ai = f"\n% Contour {contour_id} ({len(points)} points, area: {area:.1f})\n"
    ai += "newpath\n"
    ai += f"{points[0][0]} {height - points[0][1]} moveto\n"
    
    for j in range(1, len(points)):
        x, y = points[j]
        ai += f"{x} {height - y} lineto\n"
    
    ai += "closepath\n"
    ai += "stroke\n"
    return ai

def trace_lines(img, output_prefix, width, height):
    """描摹所有线条"""
    edges = detect_black_lines(img)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    total_contours = len(sorted_contours)
    total_points = sum(len(c) for c in sorted_contours)
    
    print(f"    检测到 {total_contours} 个线条轮廓, {total_points} 个点")
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Black Lines Vector - {total_contours} paths</title>
  <desc>Vector trace of black lines</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <g id="lines" stroke="black" fill="none" stroke-width="0.5">
'''
    
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Black Lines Tracer
%%Title: {output_prefix}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% BLACK LINES VECTOR
% {total_contours} contours, {total_points} points

0 setgray
1 setlinewidth

'''
    
    for i, contour in enumerate(sorted_contours):
        area = cv2.contourArea(contour)
        if area < 5:
            continue
        
        points = contour.reshape(-1, 2)
        if len(points) < 2:
            continue
        
        svg_content += contours_to_svg(points, height, i)
        ai_content += contours_to_ai(points, height, i, area)
    
    svg_content += '''  </g>
</svg>'''
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    svg_path = f"{output_prefix}.svg"
    ai_path = f"{output_prefix}.ai"
    
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    with open(ai_path, "w", encoding="utf-8") as f:
        f.write(ai_content)
    
    print(f"    生成: {svg_path}, {ai_path}")
    
    return total_contours

def trace_with_hierarchical_lines(img, output_prefix, width, height):
    """分层描摹线条（按粗细分层）"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    edge_configs = [
        (10, 30, "fine"),
        (30, 80, "medium"),
        (80, 150, "thick"),
    ]
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Hierarchical Lines Vector</title>
  <desc>Black lines traced with multiple thresholds</desc>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <defs>
    <style>
      .fine {{ stroke: #333; stroke-width: 0.3; }}
      .medium {{ stroke: #666; stroke-width: 0.5; }}
      .thick {{ stroke: #000; stroke-width: 1; }}
    </style>
  </defs>
'''
    
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Hierarchical Lines Tracer
%%Title: {output_prefix}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
>> setpagedevice

% HIERARCHICAL BLACK LINES

'''
    
    total_count = 0
    
    for threshold_low, threshold_high, style_name in edge_configs:
        edges = cv2.Canny(blurred, threshold_low, threshold_high)
        
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        svg_content += f'\n  <!-- {style_name} lines -->\n'
        svg_content += f'  <g id="{style_name}" class="{style_name}">\n'
        
        ai_content += f"\n% {style_name} lines ({len(sorted_contours)} contours)\n"
        
        for i, contour in enumerate(sorted_contours):
            area = cv2.contourArea(contour)
            if area < 3:
                continue
            
            points = contour.reshape(-1, 2)
            if len(points) < 2:
                continue
            
            total_count += 1
            
            svg_content += contours_to_svg(points, height, f"{style_name}_{i}")
            
            ai_content += contours_to_ai(points, height, total_count, area)
        
        svg_content += '  </g>\n'
    
    svg_content += '</svg>'
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    svg_path = f"{output_prefix}_hierarchical.svg"
    ai_path = f"{output_prefix}_hierarchical.ai"
    
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    with open(ai_path, "w", encoding="utf-8") as f:
        f.write(ai_content)
    
    print(f"    生成: {svg_path}, {ai_path}")
    
    return total_count

def process_pdf_lines(pdf_path="a.pdf"):
    """处理PDF黑色线条边框"""
    print("=" * 60)
    print("PDF 黑色线条描摹 (无填充，纯线条)")
    print("=" * 60)
    
    print("\n[1/3] 提取PDF页面...")
    image_path = extract_page_as_image(pdf_path, dpi=300)
    
    print("\n[2/3] 读取并处理...")
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return
    
    height, width = img.shape[:2]
    print(f"    图片尺寸: {width}x{height}")
    
    base_name = "extracted/lines"
    
    print("\n    描摹所有线条...")
    trace_lines(img, base_name, width, height)
    
    print("\n    生成分层线条...")
    trace_with_hierarchical_lines(img, base_name, width, height)
    
    print("\n[3/3] 完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - lines.svg              (SVG - 所有线条)")
    print("  - lines.ai               (AI - 所有线条)")
    print("  - lines_hierarchical.svg (SVG - 分层线条)")
    print("  - lines_hierarchical.ai  (AI - 分层线条)")
    print("\n使用说明:")
    print("  1. Adobe Illustrator: 打开 lines.ai 或 lines_hierarchical.ai")
    print("  2. 所有路径都是纯线条(stroke)，无填充(fill)")
    print("  3. 在AI中可以轻松选中并编辑每条线条")

if __name__ == "__main__":
    process_pdf_lines("a.pdf")
