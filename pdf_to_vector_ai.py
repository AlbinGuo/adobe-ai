import fitz
import cv2
import numpy as np
from PIL import Image
import os

def extract_image_from_pdf(pdf_path, output_dir="extracted"):
    """从PDF中提取图片"""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    extracted_files = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_path = os.path.join(output_dir, f"image_{page_num}_{img_index}.png")
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            extracted_files.append(image_path)
            print(f"已提取: {image_path}")
    
    doc.close()
    
    if not extracted_files:
        doc = fitz.open(pdf_path)
        if doc.page_count > 0:
            page = doc[0]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image_path = os.path.join(output_dir, "page_full.png")
            pixmap.save(image_path)
            extracted_files.append(image_path)
            print(f"已提取整个页面: {image_path}")
        doc.close()
    
    return extracted_files

def image_contours_to_svg(img_array, output_path, threshold=100, min_area=10):
    """将图片轮廓转换为SVG矢量图"""
    height, width = img_array.shape[:2]
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, threshold, threshold * 2)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Vector traced image</title>
  <desc>Auto-traced vector with {len(contours)} contours</desc>
  <g id="all-contours" fill="none" stroke="black" stroke-width="1">
'''
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        svg_content += f'    <path id="contour_{i}" d="'
        
        points = contour.reshape(-1, 2)
        if len(points) > 0:
            svg_content += f"M {points[0][0]} {points[0][1]} "
            
            for j in range(1, len(points)):
                svg_content += f"L {points[j][0]} {points[j][1]} "
            
            svg_content += "Z"
        
        svg_content += f'" />\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    print(f"已生成SVG: {output_path}")
    return len(contours)

def create_layered_svg(img_array, output_path, num_levels=8):
    """创建分层SVG（适合Adobe Illustrator编辑）"""
    height, width = img_array.shape[:2]
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    
    levels = []
    for i in range(num_levels):
        threshold = int(255 * (i + 1) / (num_levels + 1))
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        levels.append((threshold, contours))
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Layered Vector Trace</title>
  <defs>
    <style>
'''

    colors = ['#000000', '#1a1a1a', '#333333', '#4d4d4d', '#666666', '#808080', '#999999', '#b3b3b3']
    for i in range(num_levels):
        svg_content += f'      .layer-{i} {{ fill: {colors[i % len(colors)]}; stroke: none; opacity: {1 - i*0.1}; }}\n'
        svg_content += f'      .layer-{i}-stroke {{ fill: none; stroke: {colors[i % len(colors)]}; stroke-width: 0.5; }}\n'

    svg_content += '''    </style>
  </defs>
  <g id="background" fill="white">
    <rect width="100%" height="100%"/>
  </g>
'''

    for level_idx, (threshold, contours) in enumerate(levels):
        svg_content += f'  <g id="layer_{level_idx}_fill" class="layer-{level_idx}">\n'
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 5:
                continue
            
            points = contour.reshape(-1, 2)
            if len(points) > 0:
                svg_content += f'    <path id="l{level_idx}_c{i}" d="'
                svg_content += f"M {points[0][0]} {points[0][1]} "
                for j in range(1, len(points)):
                    svg_content += f"L {points[j][0]} {points[j][1]} "
                svg_content += f'Z"/>\n'
        svg_content += '  </g>\n'
        
        svg_content += f'  <g id="layer_{level_idx}_stroke" class="layer-{level_idx}-stroke">\n'
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 5:
                continue
            
            points = contour.reshape(-1, 2)
            if len(points) > 0:
                svg_content += f'    <path id="ls{level_idx}_c{i}" d="'
                svg_content += f"M {points[0][0]} {points[0][1]} "
                for j in range(1, len(points)):
                    svg_content += f"L {points[j][0]} {points[j][1]} "
                svg_content += f'Z"/>\n'
        svg_content += '  </g>\n'
    
    svg_content += '</svg>'
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    print(f"已生成分层SVG: {output_path}")

def create_ai_file(img_array, output_path, num_levels=6):
    """创建Adobe Illustrator兼容的AI文件"""
    height, width = img_array.shape[:2]
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    
    ai_content = f'''%!AI-Adobe_Illustrator-3.0
%%Creator: Python PDF to Vector Converter
%%Title: {output_path}
%%CreationDate: {__import__('datetime').datetime.now().isoformat()}
%%Pages: 1
%%BoundingBox: 0 0 {width} {height}

<<
  /PageSize [{width} {height}]
  /ViewBoxes false
>> setpagedevice

% Layered vector trace with {num_levels} levels

'''

    for level_idx in range(num_levels):
        threshold = int(255 * (level_idx + 1) / (num_levels + 1))
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ai_content += f"\n% Layer {level_idx + 1} (threshold: {threshold})\n"
        ai_content += f"/layerName (Layer_{level_idx + 1}) def\n"
        
        gray_level = 1 - (level_idx / num_levels)
        ai_content += f"{gray_level} setgray\n"
        
        contour_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue
            
            contour_count += 1
            ai_content += "\n% Contour\n"
            ai_content += "newpath\n"
            
            points = contour.reshape(-1, 2)
            if len(points) > 0:
                ai_content += f"{points[0][0]} {height - points[0][1]} moveto\n"
                
                for j in range(1, len(points)):
                    ai_content += f"{points[j][0]} {height - points[j][1]} lineto\n"
            
            ai_content += "closepath\n"
            ai_content += "fill\n"
        
        ai_content += f"% End of Layer {level_idx + 1} ({contour_count} contours)\n"
    
    ai_content += "\nshowpage\n"
    ai_content += "%%EndDocument\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ai_content)
    
    print(f"已生成AI文件: {output_path}")

def create_detailed_trace(img_array, output_path):
    """创建带详细信息的SVG"""
    height, width = img_array.shape[:2]
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = sum(cv2.contourArea(c) for c in contours)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <metadata>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:cc="http://creativecommons.org/ns#"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
      <cc:Work rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
        <dc:title>Vector traced image</dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <title>Detailed Vector Trace</title>
  <desc>Original size: {width}x{height}, Contours: {len(contours)}, Total area: {total_area:.0f} px²</desc>
  
  <g id="contours" fill="black" stroke="none">
'''

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for i, contour in enumerate(sorted_contours[:500]):
        area = cv2.contourArea(contour)
        if area < 1:
            continue
        
        opacity = min(1.0, 0.3 + (area / total_area) * 10)
        
        svg_content += f'    <path id="c{i}" data-area="{area:.1f}" data-opacity="{opacity:.3f}" d="'
        
        points = contour.reshape(-1, 2)
        if len(points) > 0:
            svg_content += f"M {points[0][0]} {points[0][1]} "
            for j in range(1, len(points)):
                svg_content += f"L {points[j][0]} {points[j][1]} "
            svg_content += "Z"
        
        svg_content += f'" fill-opacity="{opacity}"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    print(f"已生成详细SVG: {output_path}")

def process_pdf_images(pdf_path="a.pdf"):
    """处理PDF中的图片并转换为矢量格式"""
    print("=" * 60)
    print("PDF 图片转矢量图 (Adobe Illustrator 兼容)")
    print("=" * 60)
    
    print("\n[1/4] 提取PDF中的图片...")
    images = extract_image_from_pdf(pdf_path)
    
    if not images:
        doc = fitz.open(pdf_path)
        if len(doc) > 0:
            page = doc[0]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image_path = "extracted/page_full.png"
            pixmap.save(image_path)
            images = [image_path]
        doc.close()
    
    for image_path in images:
        print(f"\n[2/4] 处理图片: {image_path}")
        base_name = os.path.splitext(image_path)[0]
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            continue
        
        print(f"    图片尺寸: {img.shape[1]}x{img.shape[0]}")
        
        print("[3/4] 生成矢量图...")
        contours = image_contours_to_svg(img, f"{base_name}_contours.svg", threshold=100, min_area=5)
        print(f"    检测到 {contours} 个轮廓")
        
        create_layered_svg(img, f"{base_name}_layered.svg", num_levels=8)
        
        create_ai_file(img, f"{base_name}.ai", num_levels=6)
        
        create_detailed_trace(img, f"{base_name}_detailed.svg")
    
    print("\n[4/4] 完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  [Folder] extracted/")
    for f in sorted(os.listdir("extracted")):
        if not f.endswith('.png'):
            print(f"    - {f}")
    
    print("\n推荐使用:")
    print("  Adobe Illustrator: 打开 *.ai 文件 (支持图层)")
    print("  Inkscape: 打开 *_layered.svg 文件")
    print("  浏览器: 打开 *_detailed.svg 文件")
    print("\n提示:")
    print("  - ai 文件可直接拖入Adobe Illustrator打开")
    print("  - _layered.svg 包含8个灰度层级")
    print("  - _contours.svg 包含所有轮廓线")

if __name__ == "__main__":
    process_pdf_images("a.pdf")
