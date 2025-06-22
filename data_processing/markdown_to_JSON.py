import re
import json
import uuid

def parse_markdown_to_json(md_content):
    # Tách file theo các heading
    headings = re.split(r'^(#{1,7}\s+.+)$', md_content, flags=re.MULTILINE)
    
    # Mảng lưu trữ đường dẫn heading hiện tại tại mỗi cấp độ
    # Ví dụ: current_path[0] là heading cấp 1 hiện tại, current_path[1] là heading cấp 2 hiện tại, v.v.
    current_path = [None, None, None, None, None, None, None]
    result = []
    
    for i in range(1, len(headings), 2):
        heading = headings[i].strip()
        content = headings[i+1].strip() if i+1 < len(headings) else ""
        
        # Xác định level của heading
        level = len(re.match(r'^(#+)', heading).group(1))
        heading_text = re.sub(r'^#+\s+', '', heading)
        
        # Cập nhật đường dẫn heading
        # Quan trọng: Reset các cấp heading thấp hơn khi gặp một heading mới
        current_path[level-1] = heading_text
        for j in range(level, 7):
            current_path[j] = None
        
        # Tạo đường dẫn heading đầy đủ cho heading hiện tại
        heading_path = []
        for j in range(level):
            if current_path[j] is not None:
                heading_path.append(current_path[j])
        
        # Tách nội dung thành các phần "Điều" và văn bản thông thường
        dieu_pattern = r'(Điều\s+\d+[^:]*:[^\n]+(?:\n(?!Điều\s+\d+)[^\n]+)*)'
        
        if re.search(dieu_pattern, content):
            # Tách các phần văn bản theo "Điều"
            parts = re.split(f'({dieu_pattern})', content)
            
            for j in range(len(parts)):
                part = parts[j].strip()
                if not part:
                    continue
                
                if re.match(dieu_pattern, part):
                    # Đây là một "Điều"
                    dieu_number_match = re.search(r'Điều\s+(\d+)', part)
                    dieu_number = dieu_number_match.group(1) if dieu_number_match else "unknown"
                    
                    # Lấy 3 heading gần nhất
                    close_headings = get_closest_headings(heading_path)
                    
                    result.append({
                        "id": str(uuid.uuid4()),
                        "content": part,
                        "metadata": {
                            "heading_path": close_headings,
                            "content_type": "dieu",
                            "dieu_number": dieu_number,
                            "level": level + 1
                        }
                    })
                else:
                    # Đây là văn bản thông thường
                    # Chia thành các đoạn
                    paragraphs = re.split(r'\n\s*\n', part)
                    for paragraph in paragraphs:
                        paragraph = paragraph.strip()
                        if paragraph:
                            # Lấy 3 heading gần nhất
                            close_headings = get_closest_headings(heading_path)
                            
                            result.append({
                                "id": str(uuid.uuid4()),
                                "content": paragraph,
                                "metadata": {
                                    "heading_path": close_headings,
                                    "content_type": "paragraph",
                                    "level": level + 1
                                }
                            })
        else:
            # Không có "Điều", toàn bộ là văn bản thông thường
            if content:
                paragraphs = re.split(r'\n\s*\n', content)
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if paragraph:
                        # Lấy 3 heading gần nhất
                        close_headings = get_closest_headings(heading_path)
                        
                        result.append({
                            "id": str(uuid.uuid4()),
                            "content": paragraph,
                            "metadata": {
                                "heading_path": close_headings,
                                "content_type": "paragraph",
                                "level": level + 1
                            }
                        })
    
    return result

def get_closest_headings(heading_path, max_headings=3):
    """
    Lấy tối đa 3 heading gần nhất, bảo toàn thứ tự phân cấp
    """
    if len(heading_path) <= max_headings:
        return heading_path
    else:
        # Lấy 3 heading gần nhất (cấp thấp nhất)
        return heading_path[-max_headings:]

# Sử dụng hàm
with open('data\\Markdown\\SO_TAY_SINH_VIEN_2024-2025.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

json_data = parse_markdown_to_json(md_content)

# Ghi ra file JSON
with open('data\\JSON\\SO_TAY_SINH_VIEN.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)