import fitz
import os

def extract_vector_content(pdf_path, output_path="vector_content.pdf"):
    """提取PDF中的矢量内容并保存为新PDF"""
    doc = fitz.open(pdf_path)
    output_doc = fitz.open()
    
    for page in doc:
        rect = page.rect
        drawings = page.get_drawings()
        
        new_page = output_doc.new_page(width=rect.width, height=rect.height)
        
        for drawing in drawings:
            items = drawing.get("items", [])
            path = drawing.get("path", None)
            
            if path:
                new_page.insert_vector_path(rect=drawing.get("rect", rect), path=path, color=drawing.get("color", (0,0,0)), fill=drawing.get("fill", None))
    
    output_doc.save(output_path)
    output_doc.close()
    doc.close()
    print(f"已保存矢量内容: {output_path}")

def extract_as_svg(page, output_svg="content.svg"):
    """将PDF页面导出为SVG（保留矢量特性）"""
    svg_content = page.get_svg_image()
    
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write(svg_content)
    
    print(f"已保存SVG: {output_svg}")

def extract_as_pixmap(page, output_image="page_image.png"):
    """将PDF页面保存为图片"""
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pixmap.save(output_image)
    print(f"已保存图片: {output_image}")

def create_combined_pdf(page, output_pdf="combined_output.pdf"):
    """创建包含矢量和图片的组合PDF"""
    rect = page.rect
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1))
    
    output_doc = fitz.open()
    new_page = output_doc.new_page(width=rect.width, height=rect.height)
    new_page.insert_image(rect, pixmap=pixmap)
    
    output_doc.save(output_pdf)
    output_doc.close()
    print(f"已保存组合PDF: {output_pdf}")

if __name__ == "__main__":
    extract_vector_content("a.pdf", "vector_content.pdf")
    
    doc = fitz.open("a.pdf")
    extract_as_svg(doc[0], "content.svg")
    extract_as_pixmap(doc[0], "page_image.png")
    create_combined_pdf(doc[0], "combined_output.pdf")
    doc.close()