import fitz
import cv2
import numpy as np
from PIL import Image
import os

def extract_page_as_image(pdf_path, output_path="extracted/page_full.png", dpi=300):
    """将PDF页面提取为高分辨率图片"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    pixmap = page.get_pixmap(matrix=mat)
    pixmap.save(output_path)
    
    doc.close()
    print(f"已提取页面: {output_path} ({pixmap.width}x{pixmap.height})")
    return output_path

def advanced_trace(img):
    """高级矢量描摹 - 提取所有细节"""
    height, width = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    simplified_contours = []
    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified_contours.append(approx)
    
    return simplified_contours, width, height

def contours_to_svg_path(contour):
    """将轮廓转换为SVG路径"""
    if len(contour) == 0:
        return ""
    
    points = contour.reshape(-1, 2)
    path = f"M {points[0][0]} {points[0][1]}"
    
    for j in range(1, len(points)):
        path += f" L {points[j][0]} {points[j][1]}"
    
    path += " Z"
    return path

def create_full_trace_svg(img, output_path, simplify=True):
    """创建完整的矢量描摹SVG"""
    height, width = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    edges = cv2.Canny(blurred, 20, 60, apertureSize=3)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = sum(cv2.contourArea(c) for c in contours)
    
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:ai="http://ns.adobe.com/AdobeIllustrator/10.0/"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <metadata>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:dc="http://purl.org/dc/elements/1.1/"
             xmlns:cc="http://creativecommons.org/ns#">
      <cc:Work rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
        <dc:title>Full Vector Trace</dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <title>Complete Vector Trace - {len(contours)} contours</title>
  <desc>Original: {width}x{height}, Contours: {len(contours)}, Total Area: {total_area:.0f}px</desc>
  
  <!-- Background -->
  <rect width="100%" height="100%" fill="white"/>
  
  <!-- All paths grouped by size -->
  <g id="all_paths" fill="black" stroke="none">
'''
    
    for i, contour in enumerate(contours_sorted):
        area = cv2.contourArea(contour)
        if area < 2:
            continue
        
        if simplify:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
        
        path_data = contours_to_svg_path(contour)
        if not path_data:
            continue
        
        opacity = min(1.0, 0.1 + (area / total_area) * 5)
        
        svg_content += f'    <path id="path_{i}" data-area="{area:.1f}" data-order="{i}" '
        svg_content += f'fill-opacity="{opacity:.3f}" d="{path_data}"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    print(f"已生成完整描摹SVG: {output_path}")
    return len(contours)

def create_multi_layer_svg(img, output_path, num_layers=16):
    """创建多层次的详细描摹SVG"""
    height, width = img.shape[:2]
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        gray = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges_layers = []
    for threshold in [20, 40, 60, 80, 100, 120, 150, 180]:
        edges = cv2.Canny(blurred, threshold * 0.5, threshold)
        edges_layers.append(edges)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Multi-Layer Vector Trace</title>
  <defs>
    <style>
      .layer {{ fill: none; stroke: black; }}
'''
    
    colors = ['#000000', '#111111', '#222222', '#333333', '#444444', '#555555', '#666666', '#777777',
              '#888888', '#999999', '#AAAAAA', '#BBBBBB', '#CCCCCC', '#DDDDDD', '#EEEEEE', '#FFFFFF']
    
    for i in range(num_layers):
        svg_content += f'      .l{i} {{ stroke: {colors[i % len(colors)]}; stroke-width: {0.5 + i*0.1}; }}\n'
        svg_content += f'      .lf{i} {{ fill: {colors[i % len(colors)]}; fill-opacity: 0.3; stroke: none; }}\n'
    
    svg_content += '''    </style>
  </defs>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <!-- Combined edges as paths -->
  <g id="edges_combined">
'''
    
    for layer_idx, edges in enumerate(edges_layers):
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        svg_content += f'    <g id="edge_layer_{layer_idx}" class="layer l{layer_idx % num_layers}">\n'
        
        contour_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5:
                continue
            
            contour_count += 1
            
            if simplify_contour(contour, area) is None:
                continue
            
            path_data = contours_to_svg_path(contour)
            if not path_data:
                continue
            
            svg_content += f'      <path id="e{layer_idx}_{contour_count}" d="{path_data}"/>\n'
        
        svg_content += f'    </g>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    print(f"已生成多层SVG: {output_path}")

def simplify_contour(contour, area, factor=0.01):
    """简化轮廓"""
    if area < 10:
        return contour
    try:
        epsilon = factor * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)
    except:
        return contour

def create_complete_ai(img, output_path):
    """创建完整的Adobe Illustrator兼容文件"""
    height, width = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    edges = cv2.Canny(blurred, 20, 60)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    
    total_contours = len(contours_sorted)
    print(f"检测到 {total_contours} 个轮廓")
    
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Python Vector Tracer
%%Title: {os.path.basename(output_path)}
%%CreationDate: {__import__('datetime').datetime.now().isoformat()}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}
%%HiResBoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
  /ViewBoxes false
>> setpagedevice

%% Document Custom Data
% Total Contours: {total_contours}
% Original Image Size: {width}x{height}

'''

    ai_content += f"% AI_File = {output_path}\n"
    ai_content += f"% Contours = {total_contours}\n\n"
    
    ai_content += '''% Inner path data structure
/internaldict 256 dict def
internaldict begin
  /contours 0 def
end

'''

    contour_id = 0
    
    for i, contour in enumerate(contours_sorted):
        area = cv2.contourArea(contour)
        if area < 2:
            continue
        
        try:
            epsilon = 0.003 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
        except:
            approx = contour
        
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            continue
        
        contour_id += 1
        
        gray_value = int(255 * (1 - area / (width * height) * 100))
        gray_value = max(0, min(255, gray_value))
        
        ai_content += f"\n% Contour {contour_id} - Area: {area:.1f}\n"
        ai_content += f"% Bounds: {points[:,0].min()}-{points[:,0].max()} x {points[:,1].min()}-{points[:,1].max()}\n"
        ai_content += "newpath\n"
        
        ai_content += f"{points[0][0]} {height - points[0][1]} moveto\n"
        
        for j in range(1, len(points)):
            x, y = points[j]
            px, py = points[j-1]
            
            dx = x - px
            dy = y - py
            
            if abs(dx) > 50 or abs(dy) > 50:
                ai_content += f"{x} {height - y} moveto\n"
            else:
                ai_content += f"{x} {height - y} lineto\n"
        
        ai_content += "closepath\n"
        
        if area > 1000:
            ai_content += f"{gray_value / 255} setgray\n"
            ai_content += "fill\n"
        else:
            ai_content += f"{gray_value / 255} setgray\n"
            ai_content += "0.5 setlinewidth\n"
            ai_content += "stroke\n"
        
        if contour_id % 100 == 0:
            print(f"  已处理 {contour_id}/{total_contours} 个轮廓...")
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ai_content)
    
    print(f"已生成完整AI文件: {output_path}")

def create_trace_with_fills(img, output_path):
    """创建带有填充区域的完整描摹（适合图像描摹效果）"""
    height, width = img.shape[:2]
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_internal, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:ai="http://ns.adobe.com/AdobeIllustrator/10.0/"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Complete Image Trace with Fills</title>
  <desc>Image trace with fills and strokes - {len(contours)} external, {len(contours_internal)} total contours</desc>
  
  <metadata>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <cc:Work rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:title>Complete Vector Trace</dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  
  <defs>
    <linearGradient id="tone" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:black;stop-opacity:1" />
      <stop offset="100%" style="stop-color:white;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <!-- Group for all fill regions -->
  <g id="fills" fill="black" stroke="none">
'''

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 20:
            continue
        
        try:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
        except:
            approx = contour
        
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            continue
        
        path_data = contours_to_svg_path(contour)
        if not path_data:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1
        
        svg_content += f'    <path id="fill_{i}" data-area="{area}" data-aspect="{aspect_ratio:.2f}" d="{path_data}"/>\n'
    
    svg_content += '''  </g>
  
  <!-- Group for stroke edges -->
  <g id="edges" fill="none" stroke="#333333" stroke-width="0.5">
'''

    for i, contour in enumerate(contours_internal[:len(contours)]):
        area = cv2.contourArea(contour)
        if area < 5:
            continue
        
        path_data = contours_to_svg_path(contour)
        if not path_data:
            continue
        
        svg_content += f'    <path id="edge_{i}" d="{path_data}"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    print(f"已生成填充描摹SVG: {output_path}")

def create_ai_with_groups(img, output_path, num_groups=8):
    """创建带有分组信息的Adobe Illustrator文件"""
    height, width = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 30, 90)
    all_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_sorted = sorted(all_contours, key=cv2.contourArea, reverse=True)
    
    group_size = max(1, len(contours_sorted) // num_groups)
    
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Python Vector Tracer - Complete Trace
%%Title: {os.path.basename(output_path)}
%%CreationDate: {__import__('datetime').datetime.now().isoformat()}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
  /ViewBoxes false
>> setpagedevice

% ================================
% COMPLETE VECTOR TRACE
% ================================
% Total Contours: {len(contours_sorted)}
% Image Size: {width}x{height}
% Groups: {num_groups}

'''

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, len(contours_sorted))
        
        if start_idx >= len(contours_sorted):
            break
        
        gray_level = 1 - (group_idx / num_groups)
        
        ai_content += f"\n% ================================ GROUP {group_idx + 1} ================================\n"
        ai_content += f"% Contours {start_idx + 1} to {end_idx}\n"
        ai_content += f"/layerName (Group_{group_idx + 1}) def\n"
        ai_content += f"{gray_level} setgray\n"
        
        group_contour_count = 0
        for i in range(start_idx, end_idx):
            contour = contours_sorted[i]
            area = cv2.contourArea(contour)
            
            if area < 5:
                continue
            
            try:
                epsilon = 0.004 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
            except:
                approx = contour
            
            points = approx.reshape(-1, 2)
            if len(points) < 3:
                continue
            
            group_contour_count += 1
            
            ai_content += f"\n% Contour {i + 1} (Group {group_idx + 1})\n"
            ai_content += "newpath\n"
            
            ai_content += f"{points[0][0]} {height - points[0][1]} moveto\n"
            
            for j in range(1, len(points)):
                x, y = points[j]
                ai_content += f"{x} {height - y} lineto\n"
            
            ai_content += "closepath\n"
            
            if area > 100:
                ai_content += "fill\n"
            else:
                ai_content += "0.3 setlinewidth\nstroke\n"
        
        ai_content += f"% End Group {group_idx + 1} ({group_contour_count} contours)\n"
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ai_content)
    
    print(f"已生成分组AI文件: {output_path}")
    return len(contours_sorted)

def process_pdf_complete(pdf_path="a.pdf"):
    """完整的PDF矢量描摹流程"""
    print("=" * 70)
    print("PDF 图片完整矢量描摹")
    print("=" * 70)
    
    print("\n[1/6] 提取PDF页面为高分辨率图片...")
    image_path = extract_page_as_image(pdf_path, dpi=300)
    
    print("\n[2/6] 读取图片...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    print(f"    图片尺寸: {img.shape[1]}x{img.shape[0]}")
    
    base_name = os.path.splitext(image_path)[0]
    
    print("\n[3/6] 生成完整描摹SVG...")
    create_full_trace_svg(img, f"{base_name}_complete.svg")
    
    print("\n[4/6] 生成分层SVG...")
    create_multi_layer_svg(img, f"{base_name}_multilayer.svg", num_layers=12)
    
    print("\n[5/6] 生成Adobe Illustrator格式...")
    create_complete_ai(img, f"{base_name}_complete.ai")
    
    create_trace_with_fills(img, f"{base_name}_with_fills.svg")
    
    total_contours = create_ai_with_groups(img, f"{base_name}_grouped.ai", num_groups=12)
    
    print("\n[6/6] 完成！")
    print("=" * 70)
    print(f"\n检测到的轮廓总数: {total_contours}")
    print("\n生成的文件:")
    print(f"  - {base_name}_complete.svg  (完整SVG描摹)")
    print(f"  - {base_name}_multilayer.svg  (多层SVG)")
    print(f"  - {base_name}_with_fills.svg  (带填充SVG)")
    print(f"  - {base_name}_complete.ai  (完整AI格式)")
    print(f"  - {base_name}_grouped.ai  (分组AI格式)")
    
    print("\n使用说明:")
    print("  1. Adobe Illustrator: 打开 *_complete.ai 或 *_grouped.ai")
    print("  2. Inkscape: 打开任何 *.svg 文件")
    print("  3. _grouped.ai 包含12个分组，每个分组约100个轮廓")
    print("  4. 在AI中可以进一步使用'图像描摹'进行精细调整")

if __name__ == "__main__":
    process_pdf_complete("a.pdf")
