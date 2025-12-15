import logging
import re
from typing import Dict, Any
import json


def enhanced_json_extractor(text: str) -> Dict[str, Any]:
    """增强型 JSON 提取器，支持多种格式和错误修复"""
    # 优先级匹配模式（按顺序尝试）
    patterns = [
        (r'(?is)<Result>(.*?)</Result>', 5.0),
        (r'(?s)```json\n(.*?)```', 4.0),  # 标准 JSON 代码块
        (r'(?s)<json>(.*?)</json>', 3.5),  # XML 式包裹
        (r'(?s)```\n(.*?)```', 3.0),  # 无类型代码块
        (r'(?s)JSON:\s*({.*?})(?=\n\S+:|$)', 2.5),  # 键值对形式
        (r'(?s){.*}', 2.0)  # 裸 JSON 对象
    ]

    candidates = []

    # 多模式扫描
    for pattern, score in patterns:
        for match in re.finditer(pattern, text):
            try:
                raw_json = match.group(1).strip()
            except IndexError:
                raw_json = match.group(0).strip()
            # if not raw_json or len(raw_json) > 100:
            if not raw_json:
                continue

            # 预处理修复
            repaired = repair_json(raw_json)
            if validate_json(repaired):
                candidates.append((repaired, score + len(repaired) / 1000))
                # break
        if len(candidates) > 0:
            break

    # 按置信度排序
    candidates.sort(key=lambda x: x[1], reverse=True)

    # 尝试解析最佳候选
    for json_str, _ in candidates:
        try:
            return json.loads(json_str)
        except Exception as e:
            # if "Expecting value: line 1 " in e:

            logging.debug(f"解析失败但已通过验证的 JSON: {json_str[:200]}... 错误: {str(e)}")

    # 保底：全文扫描
    return deep_scan_json(text)


def repair_json(raw: str) -> str:
    """修复常见 JSON 格式问题"""
    repairs = [
        (r"'([^']*)'", '"\\1"'),  # 单引号转双引号
        (r",\s*}", "}"),  # 去除尾部逗号
        (r"//.*?\n", ""),  # 删除行注释
        (r"(?<!\\)\\.", ""),  # 去除非法转义
        (r"\bNaN\b", "null"),  # 处理 NaN
        (r"\bInfinity\b", "0")  # 处理无穷大
    ]

    for pattern, repl in repairs:
        raw = re.sub(pattern, repl, raw)
    return raw


def validate_json(raw: str) -> bool:
    """快速验证 JSON 结构有效性"""
    return (
            raw.startswith('{') and
            raw.endswith('}') and
            ('brand' in raw or 'model' in raw) and  # 业务字段验证
            raw.count('{') == raw.count('}')
    )


def deep_scan_json(text: str) -> Dict[str, Any]:
    """深度扫描提取 JSON 对象"""
    bracket_balance = 0
    start_index = -1
    best_candidate = ""

    for i, char in enumerate(text):
        if char == '{':
            bracket_balance += 1
            if start_index == -1:
                start_index = i
        elif char == '}':
            bracket_balance -= 1
            if bracket_balance == 0 and start_index != -1:
                candidate = text[start_index:i + 1]
                if len(candidate) > len(best_candidate):
                    best_candidate = candidate
                start_index = -1
        elif bracket_balance < 0:  # 不平衡重置
            start_index = -1
            bracket_balance = 0

    if best_candidate:
        try:
            return json.loads(repair_json(best_candidate))
        except:
            pass

    raise ValueError("无法提取有效 JSON 数据")


# 统一入口函数
def extract_json(result: str) -> Dict[str, Any]:
    """从响应文本中提取并验证 JSON 数据"""
    try:
        return enhanced_json_extractor(result)
    except Exception as e:
        logging.error(f"JSON 提取失败: {str(e)}")
        return {"error": "invalid_response"}

def normalize_newlines(text):
    """全面标准化换行符，包括\\n, \r\n, \\r\\n, \r"""
    # 先把所有字面上的 \\n 换成真正的 \n
    text = text.replace("\\n", "\n")
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\r", "\n")
    # 再把实际存在的 \r\n 也标准化
    text = re.sub(r'\r\n?', '\n', text)
    text = text.replace("\n\n", "\n")
    return text


def sanitize_network_info(text: str, brand) -> str:
    """
    移除文本中的IP地址、MAC地址等网络标识信息

    参数:
        text: 需要处理的原始文本

    返回:
        脱敏后的安全文本
    """
    # IP地址正则（支持IPv4和IPv6）
    # ipv4_pattern = r'\b(25[0-5]|2[0-4][0-9]|1[0-9]{2})\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\b'
    # ipv4_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b(?::[0-9]{1,5})?[\t\n\r ]*'
    # ipv4_pattern = r'[\n\t]*(?:\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b(?::[0-9]{1,5})?)[\n\t]*'
    # ipv4_pattern = r'([^\S\n])*(?:\b(?:(?:25[0-5]|2[0-4][0-9]|[012]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b(?::[0-9]{1,5})?)\s*'
    ipv6_pattern = r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b'  # 完整格式
    ipv6_short_pattern = r'\b(?:[A-Fa-f0-9]{1,4}(?::|::)){1,6}[A-Fa-f0-9]{1,4}\b'  # 简写格式

    # MAC地址正则（支持冒号、连字符、无分隔符格式）
    mac_pattern = r'\b(?:[0-9A-Fa-f]{2}[:-]?){5}(?:[0-9A-Fa-f]{2})\b'

    # 组合模式并执行替换
    # combined_pattern = re.compile(f'({ipv4_pattern})|({ipv6_pattern})|({ipv6_short_pattern})|({mac_pattern})',
    #                               flags=re.IGNORECASE | re.MULTILINE)
    combined_pattern = re.compile(f'({ipv6_pattern})|({ipv6_short_pattern})|({mac_pattern})',
                                  flags=re.IGNORECASE | re.MULTILINE)
    text = clean_snmp_special_banner(text)
    if text == '':
        return 'No device info in banner'

    # 去掉长度超过30的纯字母数字组合
    pattern = rf'\b[a-zA-Z0-9]{{{29 + 1},}}\b'
    text = re.sub(pattern, '', text)

    text = normalize_newlines(text)
    # if 'SNMP HOST Info' in text:
    #   text = 'SNMP HOST Info' + text.split('SNMP HOST Info')[1]
    text = re.sub('\\\\t', ' ', text)
    # text = re.sub('\\\\n', ' ', text)
    text = re.sub('\\\\r', ' ', text)
    # if brand not in ['hp', 'linksys', 'juniper', 'apklink']:
    # if brand in ["cisco"]:
    text = combined_pattern.sub('', text)
    # text = re.sub('[REDACTED]', '', text)
    # 过滤掉不可打印的字符
    # text = ''.join([c if c.isprintable() else ' ' for c in text])
    text = ''.join([c if (c.isprintable() or c == '\n') else ' ' for c in text])
    telnet_pattern = re.compile(r'(\\x[0-9a-fA-F+-]{2})')

    # 去除控制序列并解码为字符串
    text = telnet_pattern.sub(' ', text)

    # 移除无意义TCP条目
    text = re.sub(r'\n\d+\s+22\s+\d+\s+\d+\s+finWait1', '', text)

    # 清理空ARP记录
    text = re.sub(r'ip\s+mac地址\s+(\n\d+\s*)+', '', text)

    text = re.sub(r'\b(?:(?:[A-Z0-9]{18,})|(?:[a-z0-9]{18,}))\b', '', text)
    text = re.sub(r'[\t ]+', ' ', text).strip()
    # 保留关键UDP端口
    # text = re.sub(r'\n(\d+\s+)(?!123|162|161)\d+', '', text)

    # 处理cisco的特殊banner(列表形式，内嵌字典)


    return text


def clean_snmp_special_banner(text):
    # 处理snmp的特殊banner(列表形式，内嵌字典)
    if text.startswith('[{'):
        if "@@@@@" in text:
            text = text.split("@@@@@")[0]
        if text.endswith('}]'):
            text = json.loads(text)
            tmp = []
            for item in text:
                if item["req_name"] in ["snmp_host", "deep_scan_upnp_req"]:
                    try:
                        tmp.append(item["decode_banner"])
                    except KeyError:
                        tmp.append(item["banner"])

            if len(tmp) == 1:
                text = tmp[0]
            elif len(tmp) > 1:
                text = '$$$'.join(tmp)
            else:
                text = ''
    if text.startswith('SNMP ') or text.startswith('snmp '):
        tmp = text.split('\n\n\r\n')
        for tm in tmp:
            if tm.startswith("SNMP HOST Info") or tm.startswith("snmp host info"):
                text = tm
    return text


def remove_ntr(text):
    """
    移除\n\r\t等制表符
    :param text:
    :return:
    """
    text = ''.join([c if (c.isprintable()) else ' ' for c in text])
    text = re.sub('\\\\t', ' ', text)
    # text = re.sub('\\\\n', ' ', text)
    text = re.sub('\\\\r', ' ', text)
    text = re.sub('\r', ' ', text)
    text = re.sub('\t', ' ', text)
    # text = re.sub(r'\n{2,}', '\n', text)
    # text = re.sub(r'[\n\t]', '', text)
    text = re.sub(r'[\t\r\[\]]+', ' ', text)  # 替换制表符和回车为空格
    # text = re.sub(r'\n{3,}', '\n\n', text)  # 多换行压缩为双换行
    text = re.sub(r' +', ' ', text)  # 多空格压缩为单空格
    return text


def extract_values(text):
    # 去除包装符号和空白字符
    # json_str = text.strip('<>/ \n').replace('\'','\"')
    #
    # # 解析JSON
    # try:
    #     data = json.loads(json_str)
    # except json.JSONDecodeError as e:
    match = re.search(r'\{(.*?)\}', text, re.DOTALL)
    if match:
        json_str = '{' + match.group(1) + '}'
        json_str = json_str.replace('\'', '\"')

        # 转换单引号并加载为字典
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return {}
    else:
        if "brand" in text:
            json_str = f'{{{text.strip()}}}'
            try:
                sstr = json.loads(json_str)
                return sstr
            except json.JSONDecodeError as e:
                print("未找到有效的 JSON 结构", e)
                return {}
        if 'brand' in text:
            text = text.replace("'", "\"")
            json_str = f'{{{text.strip()}}}'
            try:
                sstr = json.loads(json_str)
                return sstr
            except json.JSONDecodeError as e:
                print("未找到有效的 JSON 结构", e)
                return {}

        print("未找到有效的 JSON 结构")
        return {}
    return data


def split_sample_to_chunks(banner, window_size=2500, overlap_size=100, min_chunk=50):
    """
    将样本切分为多个块
    :param banner:
    :return:
    """
    WINDOW_SIZE = window_size  # 窗口长度
    OVERLAP_SIZE = overlap_size  # 重叠区域
    MIN_CHUNK = min_chunk  # 最小处理块

    # 生成滑动窗口分块
    chunks = []
    if len(banner) <= WINDOW_SIZE:
        # 直接处理短文本
        chunks = [banner]
    else:
        # chunks.append(banner[:WINDOW_SIZE])  # 先加入第一分段
        # start = WINDOW_SIZE - OVERLAP_SIZE
        start = 0
        # end = 0
        while start < len(banner):
            end = min(start + WINDOW_SIZE, len(banner))
            chunk = banner[start:end]

            # 智能后移确保句子完整性
            if end < len(banner):
                split_chunk = banner[end - OVERLAP_SIZE:end]  # 重叠区域
                last_newline = split_chunk.rfind('\n')
                if last_newline < 0:
                    last_newline = split_chunk.rfind('>')
                    if last_newline < 0:
                        last_newline = split_chunk.rfind(' ')

                # 在重叠区域找最后一个换行作为上一个块的结束
                first_newline = split_chunk.find('\n')  # 在重叠区域找第一个换行作为下一个块的开始
                if first_newline < 0:
                    first_newline = split_chunk.find('<')
                    if first_newline < 0:
                        first_newline = split_chunk.find(' ')
                start_new = end - OVERLAP_SIZE + first_newline
                end = end - OVERLAP_SIZE + last_newline + 1
                chunk = banner[start:end]
                start = start_new
                # last_newline = chunk.rfind('\n')
                # if last_newline > WINDOW_SIZE - 200:  # 在结尾200字符内找换行
                #     end = start + last_newline + 1
                #     chunk = banner[start:end]
            else:
                start = end
            chunks.append(chunk)
            # start = end - OVERLAP_SIZE  # 设置重叠区域
    return chunks


def score_extracted_keywords(banner, combine_result, score_plus_item=[]):
    """
    对从banner中提取的品牌、型号、固件版本进行重新打分，考虑多次出现加分。

    Args:
        banner (str): 输入的banner文本。
        combine_result (dict): 提取结果，格式如 {"brand": [["ubiquiti", 1.0]], ...}
        score_plus_item (list): 需要加分的项
    Returns:
        dict: 更新后的combine_result，包含新的打分。
    """
    # 定义关键词列表和重复项加分规则
    base_score_plus = 3
    keyword_rules = {
        "brand": {
            "keywords": ["brand", "manufacturer", "vendor", "title", "companyname"],
            "repeat_bonus": {"brand": 1, "manufacturer": 1, "vendor": 1, "title": 0.2, "companyname": 1},
            "stop_words": ["unknown", "generic", "recording", "linux"],
            "stop_word_penalty": -5
        },
        "model": {
            "keywords": ["model", "device", "product", "title", "prod", "welcome"],
            "repeat_bonus": {"model": 3, "device": 0.15, "product": 3, "platform": 0.08, "title": 0.5, "prod": 1},
            "stop_words": ["unknown", "generic", "device",  "linux", "vt100", "notfound"],
            "alphanumeric_bonus": 1,
            "stop_word_penalty": -5
        },
        "version": {
            "keywords": ["firmwareversion", "firmware", "version", "v=", " build", "ver=", "fw ", "fmwver"],
            "repeat_bonus": {"firmwareversion": 4, "firmware": 3, "version": 1, "v=": 0.15, "build": 0.1, "fmwver": 4},
            "separator_bonus": 0.7
        }
    }

    # 多次出现加分
    multiple_appearance_bonus = 0.0  # 每次额外出现加0.1分

    # 初始化输出结果
    scored_result = {"brand": [], "model": [], "version": []}

    # 转换为小写以便匹配
    banner_lower = banner.lower()

    for key in ["brand", "model", "version"]:
        # 获取提取结果
        try:
            for [keyword, _] in combine_result[key]:
                # if len(keyword) < 3:
                #     continue
                if keyword.isdigit() and len(keyword) < 4:
                    continue
                if "xxx" in keyword or keyword == "":
                    continue
                # if (key == "model" or key == "version") and len(keyword) < 3:
                #     continue
                total_score = 0.0
                if keyword in score_plus_item:
                    total_score += base_score_plus
                # 查找关键词在banner中的所有位置
                keyword_lower = keyword.lower()
                positions = [m.start() for m in re.finditer(re.escape(keyword_lower), banner_lower)]

                # 计算每次出现的分数
                for idx, pos in enumerate(positions):
                    score = 0.0

                    # 1. 上下文关键词打分
                    # 提取前后50个字符的上下文
                    start = max(0, pos - 50)
                    end = min(len(banner), pos + len(keyword) + 50)
                    context = banner_lower[start:end]

                    # 检查关键词列表
                    for ctx_keyword in keyword_rules[key]["keywords"]:
                        # 计算关键词出现次数
                        count = len(re.findall(r'\b' + re.escape(ctx_keyword.lower()) + r'\b', context))
                        if count > 0:
                            # 基础加分（每次出现加0.1）
                            score += 0.1 * count
                            # 重复项加分
                            if ctx_keyword in keyword_rules[key]["repeat_bonus"]:
                                score += keyword_rules[key]["repeat_bonus"][ctx_keyword] * (count)

                    # 2. 格式与内容打分
                    if key == "brand":
                        if keyword.lower() in keyword_rules["brand"]["stop_words"]:
                            score += keyword_rules["brand"]["stop_word_penalty"]
                    if key == "version":
                        # 检查是否包含分隔符（. 或 -）
                        if re.search(r'[.-]', keyword):
                            score += keyword_rules["version"]["separator_bonus"]

                    elif key == "model":
                        # 检查是否为字母数字组合
                        if re.match(r'^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9-]+$', keyword):
                            score += keyword_rules["model"]["alphanumeric_bonus"]
                        # 检查停止词
                        if keyword.lower() in keyword_rules["model"]["stop_words"]:
                            score += keyword_rules["model"]["stop_word_penalty"]

                    # 3. 多次出现加分（从第二次出现开始）
                    if idx > 0:
                        score += multiple_appearance_bonus

                    # 确保单次分数非负
                    score = max(0.0, score)
                    total_score += score
                if key == "brand" and len(keyword) < 2:
                    total_score = total_score * 0.5
                # 添加到结果
                exist_flag = False  # 如果有重复，分数累加
                for item in scored_result[key]:
                    if item[0] == keyword:
                        item[1] += total_score
                        exist_flag = True
                        break
                if not exist_flag:
                    scored_result[key].append([keyword, round(total_score, 2)])  # 保留两位小数
        except Exception as e:
            print(f"Error processing keyword {key}: {e}")
            return combine_result, combine_result
    # 选择得分最高的结果（得分相同选最长的）
    final_result = {}
    for key in ["brand", "model", "version"]:
        if scored_result[key]:
            # 按分数降序排序，相同分数按长度降序
            best_result = max(
                scored_result[key],
                key=lambda x: (x[1], len(x[0]))  # 先比较分数，再比较长度
            )
            final_result[key] = best_result[0]  # 只保留关键词
        else:
            final_result[key] = ""  # 如果没有结果，设为None

    return scored_result, final_result


def parse_http_banner(banner: str):
    """提取 HTTP 状态码和响应头"""
    match = re.search(r'HTTP/\d\.\d\s+(\d{3})', banner)
    if not match:
        return None, {}
    status_code = int(match.group(1))
    http_start = match.start()
    header_end = banner.find("\r\n\r\n", http_start)
    if header_end == -1:
        header_end = len(banner)
    header_block = banner[http_start:header_end]
    headers = {}
    for line in header_block.split("\r\n")[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return status_code, headers


def is_banner_useful(banner: str):
    """根据状态码和头部特征判断是否可能含设备信息"""
    status_code, headers = parse_http_banner(banner)
    if status_code is None:
        return False

    # 无设备信息的状态码
    if 100 <= status_code < 200 or status_code in {204, 304, 400, 408, 429} or 500 <= status_code < 600:
        return False

    # 3xx 重定向，条件保留
    if 300 <= status_code < 400:
        if any(h in headers for h in ['server', 'www-authenticate']):
            return True
        return False

    # 401/403/404/405 默认保留，Server 或 WWW-Authenticate 可进一步分析
    if status_code in {401, 403, 404, 405}:
        return True

    # 2xx 默认保留
    if 200 <= status_code < 300:
        return True

    # 其他情况
    return False


if __name__ == "__main__":
    b = "SIP/2.0 200 OK\\r\\nVia: SIP/2.0/TCP 212.83.190.55:44468;branch=z9hG4bK-7592\\r\\nFrom: <sip:349@212.83.190.55;transport=TCP>;tag=160\\r\\nTo: \"Sara Razavi Zand\" <sip:989@99.116.10.51;transport=TCP>;tag=77D3F313-513744FA\\r\\nCSeq: 894 OPTIONS\\r\\nCall-ID: 760\\r\\nContact: <sip:17139369406@99.116.10.51:5060;transport=tcp>\\r\\nAllow: INVITE, ACK, BYE, CANCEL, OPTIONS, INFO, MESSAGE, SUBSCRIBE, NOTIFY, PRACK, UPDATE, REFER\\r\\nSupported: 100rel,replaces,100rel,timer,replaces,norefersub\\r\\nUser-Agent: PolycomSoundPointIP-SPIP_321-UA/4.0.15.1009\\r\\nAccept-Language: en-us,en;q=0.9\\r\\nAccept: application/sdp,text/plain,message/sipfrag,application/dialog-info+xml\\r\\nAccept-Encoding: identity\\r\\nContent-Length: 0\\r\\n\\r\\n"
    print(sanitize_network_info(b, "Polycom"))