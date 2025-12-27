代码复现
1.研究目标（Objective）
本代码旨在复现论文Word Embeddings Track Social Group Changes Across 70 Years in China中提出的历时性词向量分析方法，揭示中国社会七十年间性别角色（男性、女性）的社会认知变迁。
本代码复现论文中提出的三层分析框架：
年度词向量模型（1950–2019）：捕捉细粒度语义变迁；
Analysis 1：性别效价不对称（Gender Valence Asymmetry）;
Analysis 2：社会群体表征的时间演变（Temporal evolution of social group representations);
Analysis 3：通过逐年嵌入量化社会变迁（Quantifying social change through year-to-year embeddings）。


2. 快速开始（Quick Start）
(1)环境设置
Python版本3.8+
Bash
pip install jieba gensim pandas numpy tqdm scikit-learn matplotlib


（2）完整流程
准备数据
将1950–2019年《人民日报》按年份存为 D:/人民日报年份数据/{year}.jsonl.gz。

关键参数对齐论文：
模型：Skip-gram (sg=1)
维度：300
窗口：3
最小词频：10
分词：jieba + 自定义政治/性别关键词增强

训练年度模型（1950–2019）
Bash
python train_yearly_sg300w3.py
→ 输出：models_yearly_sg300w3/{1950..2019}.model

训练十年期模型（1950s–2010s）并生成 Table 2
Bash
python train_decade_sg300w3_and_table2.py
→ 输出：models_decade_sg300w3/*.model + table2_reproduction_*.csv

复现 Analysis 1：性别效价不对称
Bash
python analysis1_gender_valence_replication.py
→ 输出：analysis1_gender_valence_replication.csv

复现 Analysis 2：社会群体表征的时间演变(因只分析男女性别这一个维度所以数据同 Analysis 1）

复现 Analysis 3：通过逐年嵌入量化社会变迁
Bash
编辑
python analysis3_historical_events_replication.py
→ 输出： analysis3_yearly_stability.csv 和 analysis3_temporal_correlation.png


3.  核心文件说明（Variables）
文件                                         功能
train_yearly_sg300w3.py	                     训练1950–2019共70个年度Skip-gram模型（sg=1, dim=300, win=3, min_count=10）
train_decade_sg300w3_and_table2.py	         训练7个十年期模型 + 生成 Table 2（女性Top 10特质）
trait_cn.txt	                               中文人格特质词典（含效价评分），用于Analysis 1


4. 结果示例
Analysis 1 输出
加载中文人格特质词典 (trait_cn.txt)...
成功加载 464 个人格特质词
处理 1950s...
处理 1960s...
处理 1970s...
处理 1980s...
处理 1990s...
处理 2000s...
处理 2010s...

================================================================================
复现论文 Analysis 1: 性别效价不对称 (基于 AggDiffMAC)
================================================================================
Category   Group      Valence    Top 10 Traits
--------------------------------------------------------------------------------
Gender     woman      1.33       天真无邪, 纯情, 多情, 纯真, 可爱, 善良, 赤子, 倔强, 迷人, 喜乐
Gender     man        -0.10      敏捷, 凶猛, 横冲直撞, 威风凛凛, 内向, 彻底, 薄弱, 稳妥, 痛快, 吹牛

详细结果已保存至: C:\Users\Administrator\analysis1_gender_valence_replication_AggDiffMAC.csv

时期	                               复现结果	                                                论文预期
1950s–1960s	                         女性效价 >> 男性（0.94 > 0.53; 0.73 > 0.36）	            ✅ 女性被赋予高度正面道德特质（如“贤淑”、“天真”）
1970s	                               男性效价反超（0.60 < 0.81）	                            ✅ 文革时期“去性别化”，强调“革命性”、“上进”等中性/男性化特质
1990s–2000s                        	 男性效价为负（-0.42, -0.10）	                            ✅ 改革开放后，男性关联更多负面词（“冒失”、“邋遢”、“心狠手辣”）
2010s	                               两性趋近（0.32 vs 0.25）	                                ✅ 当代性别话语复杂化，传统与现代特质混合

Analysis 2 输出：
加载中文人格特质词典...
加载并清洗后：455 个人格特质词
 1950s | 女: 0.93 | 男: 0.53
 1960s | 女: 0.73 | 男: 0.36
 1970s | 女: 0.60 | 男: 0.81
 1980s | 女: 0.36 | 男: 0.35
 1990s | 女: 0.35 | 男: -0.42
 2000s | 女: 0.29 | 男: -0.10
 2010s | 女: 0.32 | 男: 0.25

==========================================================================================
复现论文 Analysis 2: Temporal Evolution of Gender Representations (1950s–2010s)

==========================================================================================
Decade Female_Valence Male_Valence                          Female_Top_Traits                              Male_Top_Traits Valence_Diff
 1950s           0.94         0.53 天真无邪, 怕羞, 厚道, 孝顺, 懂事, 娇生惯养, 淘气, 骁勇, 羞涩, 贞洁   天真无邪, 怕羞, 厚道, 骁勇, 腼腆, 懂事, 娇生惯养, 害羞, 羞涩, 怕事         0.41
 1960s           0.73         0.36 天真无邪, 羞涩, 文静, 娇生惯养, 懂事, 倔强, 腼腆, 多情, 耿直, 侠义 天真无邪, 羞涩, 娇生惯养, 淘气, 腼腆, 心不在焉, 文静, 侠义, 豪爽, 怕羞         0.37
 1970s           0.60         0.81   淘气, 文静, 健谈, 娇生惯养, 腼腆, 懂事, 厚道, 豪爽, 倔强, 感伤     淘气, 腼腆, 娇生惯养, 文静, 健谈, 上进, 懂事, 厚道, 机灵, 冒昧        -0.21
 1980s           0.36         0.35 天真无邪, 贤淑, 伶俐, 纯情, 贞洁, 邋遢, 木讷, 文静, 腼腆, 娇生惯养     天真无邪, 木讷, 贤淑, 文静, 腼腆, 伶俐, 好胜, 邋遢, 贞洁, 心软         0.01
 1990s           0.35        -0.42 贤淑, 天真无邪, 害羞, 文静, 娇生惯养, 邋遢, 胆小, 善良, 任性, 仁厚     害羞, 冒失, 邋遢, 贤淑, 文静, 好胜, 木讷, 胆小, 淘气, 娇生惯养         0.77
 2000s           0.29        -0.10 怕羞, 贤淑, 羞涩, 娇生惯养, 机灵, 害羞, 天真无邪, 邋遢, 善良, 木讷   怕羞, 贤淑, 机灵, 娇生惯养, 羞涩, 倔强, 木讷, 懂事, 邋遢, 心狠手辣         0.39
 2010s           0.32         0.25 贤淑, 天真无邪, 娇纵, 淘气, 懂事, 娇生惯养, 亲热, 腼腆, 害羞, 好胜   贤淑, 娇纵, 腼腆, 天真无邪, 好胜, 淘气, 懂事, 心软, 害羞, 娇生惯养         0.07

==========================================================================================
自动结论（基于复现结果）

==========================================================================================
• 但至2010s已回落至0.32。
• 男性在1950s效价为正（0.53），但1960s转负。
结果已保存至: C:\Users\Administrator\analysis2_gender_temporal_evolution.csv


Analysis 3 输出：
加载 465 个人格特质词
成功加载 70 年的女性向量，70 年的男性向量
相邻年份平均相关性（女性）:
  Lag 1: 0.857
  Lag 2: 0.841
  Lag 5: 0.810
  Lag 10: 0.786
女性表征相关性对比:
  1950–1965（正常期）: 0.798
  1966–1976（文革）   : 0.710  ← 应显著更低
  1977–1986（后文革）: 0.852
热力图：
  <img width="1442" height="569" alt="图片" src="https://github.com/user-attachments/assets/ebe82811-dd3a-4cac-a713-f6df2a67ca69" />

1. 研究目标（Objective）

本代码旨在复现论文《Word Embeddings Track Social Group Changes Across 70 Years in China》中提出的历时性词向量分析方法，揭示中国社会七十年间性别角色（男性、女性）的社会认知变迁。

本代码复现论文中提出的三层分析框架：

    年度词向量模型（1950-2019）：训练70个年度词向量模型，捕捉细粒度语义变迁

    十年期词向量模型（1950s-2010s）：训练7个十年期模型，分析长期趋势

    多层分析框架：

        Analysis 1：性别效价不对称（Gender Valence Asymmetry）

        Analysis 2：社会群体表征的时间演变（Temporal evolution of social group representations）

        Analysis 3：通过逐年嵌入量化社会变迁（Quantifying social change through year-to-year embeddings）

2. 快速开始（Quick Start）
环境设置

Python版本: 3.8+
bash

# 安装核心依赖包
pip install jieba gensim pandas numpy tqdm scikit-learn matplotlib
pip install datasets  # 用于加载JSONL格式数据
pip install seaborn   # 用于可视化（可选）

数据准备

    人民日报数据：

        将1950-2019年《人民日报》数据按年份存储为JSONL压缩格式

        路径：D:/人民日报年份数据/{year}.jsonl.gz

        格式示例：每个文件包含text字段的JSONL格式文本

    人格特质词典：

        路径：C:/Users/Administrator/Desktop/数据复现数据/trait_cn_cleaned.txt

        格式：每行包含"特质词 效价值"（空格分隔）

完整复现流程
步骤1：数据整理
python

# 如果数据分散在不同位置，可使用提供的文件转移脚本
# 将下载的年份数据移动到统一目录
python organize_data.py

步骤2：训练年度模型（1950-2019）
python

# 运行年度模型训练脚本（修复版）
# 参数：CBOW, dim=100, window=8, min_count=5
python train_yearly_models_fixed.py

输出：

    models_yearly_fixed/目录下的70个年度模型文件

    每个模型文件约500-800MB，总大小约35-56GB

步骤3：训练十年期模型（1950s-2010s）
python

# 运行十年期模型训练脚本
# 参数：CBOW, dim=100, window=8, min_count=5
python train_decade_models_fixed.py

输出：

    models_decade_fixed/目录下的7个十年期模型文件

    每个模型文件约1-2GB

步骤4：训练严格对齐论文的1950年模型（可选）
python

# 严格对齐论文参数：Skip-gram, dim=300, window=3, min_count=10
python train_1950_only.py

输出：

    models_yearly_fixed/1950.model（严格对齐版本）

步骤5：复现Analysis 1 - 性别效价不对称
python

# 计算AggDiffMAC并生成性别效价对比
python analysis1_gender_valence_replication.py

输出：

    analysis1_gender_valence_replication_AggDiffMAC.csv

    控制台输出最女性化和最男性化的Top 10特质

步骤6：复现Analysis 2 - 时间演变分析
python

# 计算各十年期性别特质效价
python analysis2_gender_diffmac_bidirectional.py

输出：

    analysis2_gender_diffmac_bidirectional.csv

    各十年期男女Top 10特质及平均效价

步骤7：复现Analysis 3 - 逐年变迁分析
python

# 计算逐年女性效价变化，识别关键历史时期
python analysis3_female_valence_yearly.py

输出：

    analysis3_female_valence_yearly.csv

    关键历史时期（1960, 1965, 1966, 1970, 1976, 1980, 2000, 2010）的效价变化

3. 核心文件说明
模型训练文件
文件	                       功能                             	关键参数	                                  输出
train_yearly_models_fixed.py	训练1950-2019共70个年度词向量模型	sg=0(CBOW), dim=100, window=8, min_count=5	models_yearly_fixed/{year}.model
train_decade_models_fixed.py	训练7个十年期模型（1950s-2010s）	sg=0(CBOW), dim=100, window=8, min_count=5	models_decade_fixed/{decade}.model
train_1950_only.py	严格对齐论文训练1950年模型	sg=1(Skip-gram), dim=300, window=3, min_count=10	models_yearly_fixed/1950.model
分析脚本
文件	                                  功能	                               输出
analysis1_gender_valence_replication.py	性别效价不对称分析（基于AggDiffMAC）	最女性化/男性化的Top 10特质及效价
analysis2_gender_diffmac_bidirectional.py	各十年期性别特质效价对比	各十年期男女Top 10特质及平均效价
analysis3_female_valence_yearly.py	逐年女性效价变化分析	逐年女性效价及关键时期变化
数据文件
文件	                            描述	                          格式
人民日报年份数据/{year}.jsonl.gz	1950-2019年《人民日报》文本数据	JSONL压缩格式
trait_cn_cleaned.txt	中文人格特质词典（清洗后）	每行：特质词 效价值
trait_cn.txt	原始中文人格特质词典	每行：特质词 效价值
辅助脚本
文件	            功能
数据组织相关代码	转移年份数据文件到统一目录
模型验证代码	验证模型训练效果，检查关键词相似性
4. 关键代码模块详解
4.1 分词与预处理
python

def clean_and_tokenize(text):
    """文本清洗与分词函数"""
    # 1. 去除非中文、字母、数字、基本标点
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9。，！？；：\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # 2. 按句分割
    sentences = re.split(r"[。！？\n]", text)
    
    # 3. jieba分词，过滤短词
    tokenized = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 5:
            continue
        words = jieba.lcut(sent)
        words = [w for w in words if w.strip() and len(w) >= 1]
        if len(words) >= 3:
            tokenized.append(words)
    
    return tokenized

4.2 自定义词典增强
python

# 添加关键社会历史词汇（防止被错误切分）
SOCIAL_TERMS = [
    "拥军优属", "工农兵", "地主", "资本家", "妇女", "少数民族", 
    "劳动模范", "人民公社", "合作社", "土改", "翻身", "阶级",
    "帝国主义", "苏联", "斯大林", "新民主主义", "解放区"
]   

# 添加人格特质词汇
PERSONALITY_TRAITS = [
    "天真", "活泼", "温柔", "可爱", "文静", "善良", "优雅",
    "务实", "果断", "实际", "坚强", "理智", "勇敢", "稳重"
]

for term in SOCIAL_TERMS + PERSONALITY_TRAITS:
    jieba.add_word(term, freq=2000, tag='n')

4.3 词向量模型训练
python

# 年度模型参数（修复版）
model = Word2Vec(
    sentences=all_sentences,
    vector_size=100,      # 降低维度，提升训练效率和稳定性
    window=8,             # 扩大上下文窗口
    min_count=5,          # 关键：允许低频词进入vocab
    sg=0,                 # 改用CBOW（更适合学习"群体-特质"共现）
    workers=min(6, os.cpu_count()),
    epochs=10,            # 增加迭代次数
    seed=42
)

4.4 性别效价计算
python

def compute_diffmac(model, female_labels, male_labels, trait_word):
    """计算DiffMAC = MAC(Female, w) - MAC(Male, w)"""
    mac_f = compute_mac(model, female_labels, trait_word)
    mac_m = compute_mac(model, male_labels, trait_word)
    
    if mac_f is None or mac_m is None:
        return None
    return mac_f - mac_m

def compute_mac(model, group_labels, trait_word):
    """计算群体标签与特质词的MAC（Mean Average Cosine Similarity）分数"""
    sims = []
    for label in group_labels:
        if label in model.wv and trait_word in model.wv:
            sims.append(model.wv.similarity(label, trait_word))
    return np.mean(sims) if sims else None

5. 结果示例
5.1 Analysis 1 输出
text

================================================================================
复现论文 Analysis 1: 性别效价不对称 (基于 AggDiffMAC)
================================================================================
Category   Group      Valence    Top 10 Traits
--------------------------------------------------------------------------------
Gender     woman      1.33       天真无邪, 纯情, 多情, 纯真, 可爱, 善良, 赤子, 倔强, 迷人, 喜乐
Gender     man        -0.10      敏捷, 凶猛, 横冲直撞, 威风凛凛, 内向, 彻底, 薄弱, 稳妥, 痛快, 吹牛

关键发现：

    女性关联特质多为正面（效价1.33）

    男性关联特质混合正负（效价-0.10）

    复现结果与论文趋势一致：女性被赋予更多正面道德特质

5.2 Analysis 2 输出
text

==========================================================================================
Analysis 2: Decade-wise Valence of Gender-Stereotypical Traits (by DiffMAC)
==========================================================================================
Decade Female_Valence Male_Valence                          Female_Top_Traits                              Male_Top_Traits
 1950s           1.25         0.50 多情, 幽雅, 天真无邪, 淳朴, 恬静, 纯真, 喜乐, 优雅, 可敬, 娇生惯养     灵活, 墨守成规, 独裁, 实干, 稳妥, 高明, 自满, 薄弱, 稳定, 可靠
 1960s           1.76        -0.24 淳朴, 纯朴, 纯真, 孝顺, 德高望重, 天真无邪, 多情, 可爱, 可敬, 活泼     盲目, 单纯, 厉害, 背信弃义, 主观, 拖延, 自满, 薄弱, 缄默, 高明
 1970s           0.50         0.77   可敬, 叱咤风云, 纯朴, 滑稽, 博爱, 倔强, 可恨, 刚强, 仁爱, 变态       激进, 礼貌, 独立, 无理, 主观, 贤淑, 可靠, 克制, 友好, 内向
 1980s           1.58        -0.31 纯真, 善良, 赤子, 可敬, 忠贞, 天真无邪, 淳朴, 侠义, 痴情, 忧国忧民 好大喜功, 落后, 井底之蛙, 慎重, 稳定, 墨守成规, 谨慎, 激进, 薄弱, 稳妥
 1990s           1.59        -0.46   纯情, 赤子, 喜乐, 纯真, 友爱, 无悔, 仁爱, 活泼, 天真无邪, 侠义   吓人, 小心, 谨慎, 急躁, 犹豫不决, 抖擞精神, 凶狠, 自负, 武断, 厉害
 2000s           1.77        -0.88   阳光, 体贴入微, 体贴, 积极, 可敬, 赤子, 认真, 淳朴, 纯真, 文明   厉害, 迟疑, 吹牛, 心不在焉, 吓人, 武断, 性急, 奇怪, 莽撞, 横冲直撞
 2010s           1.76        -0.37     赤子, 祥和, 好客, 阳光, 关心, 喜乐, 快乐, 体面, 积极, 淳朴     心急, 莽撞, 谨慎, 厉害, 冒失, 高明, 吓人, 横冲直撞, 小心, 凶猛

时间趋势分析：

    1950s-1960s：女性效价显著高于男性，反映传统性别观念

    1970s：文革时期男性效价反超（0.77 > 0.50），"去性别化"特征明显

    1980s-2000s：改革开放后女性效价回升，男性效价转为负值

    2010s：性别观念趋于复杂，但女性仍保持较高正面效价

5.3 Analysis 3 输出
text

=== 关键年代女性效价 ===
1960: 1.29 | 奢侈, 淳朴, 忠贞, 多情, 真实, 洞察, 忠义, 活泼, 仁厚, 可敬...
1965: 1.18 | 纯朴, 纯真, 腼腆, 善良, 叱咤风云, 刚强, 细腻, 小气, 羞涩, 正气凛然...
1966: 0.04 | 刚毅, 虚伪, 下流, 朝气, 冲动, 迷人, 真实, 卑鄙, 刚强, 叛逆...
1970: 1.89 | 刚强, 可敬, 从容, 叱咤风云, 大义凛然, 贤良, 忠义, 正气, 果敢, 机智...
1976: 0.47 | 自满, 高明, 模范, 实在, 积极, 能干, 悲观, 单纯, 不切实际, 糊涂...
1980: 1.34 | 善良, 纯朴, 活泼, 含蓄, 独特, 淳朴, 肮脏, 正直, 纯洁, 纯真...
2000: 1.99 | 快乐, 伟大, 优秀, 关心, 独特, 积极, 文明, 模范, 非凡, 创造力...
2010: 1.47 | 积极, 努力, 关心, 道德, 文明, 创造力, 矛盾, 突出, 稳妥, 阳光...

历史事件关联：

    1966年（文革开始）：女性效价骤降至0.04，关联特质出现"虚伪"、"卑鄙"等负面词

    1970年（文革中期）：效价回升至1.89，但特质转向"刚强"、"叱咤风云"等中性/男性化词汇

    1976年（文革结束）：效价降至0.47，反映社会价值观念的混乱

    2000年（新世纪）：效价达峰值1.99，女性特质高度正面化

    2010年（当代）：效价稳定在1.47，特质更加多元复杂

6. 复现结果与论文预期对比
时期	复现结果	论文预期	一致性
1950s-1960s	女性效价 >> 男性（1.25 > 0.50; 1.76 > -0.24）	女性被赋予高度正面道德特质	✅ 高度一致
1970s	男性效价反超（0.50 < 0.77）	文革时期"去性别化"，强调"革命性"特质	✅ 趋势一致
1980s	女性效价回升（1.58），男性效价为负（-0.31）	改革开放初期性别观念开始变化	✅ 基本一致
1990s-2000s	男性效价持续为负（-0.46, -0.88）	改革开放深化，男性关联更多负面词	✅ 高度一致
2010s	女性效价仍高（1.76），男性效价仍负（-0.37）	当代性别话语复杂化，传统与现代混合	✅ 趋势一致
7. 配置与参数调整
7.1 关键路径配置
python

# 数据路径配置（根据实际情况修改）
DATA_ROOT = r"D:/人民日报年份数据"  # 人民日报数据目录
MODEL_DIR = "models_yearly_fixed"   # 年度模型输出目录
DECADE_MODEL_DIR = "models_decade_fixed"  # 十年期模型输出目录
TRAIT_FILE = r"C:/Users/Administrator/Desktop/数据复现数据/trait_cn_cleaned.txt"  # 特质词典


