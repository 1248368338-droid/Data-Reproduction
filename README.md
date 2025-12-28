论文复现：历时性词向量追踪中国社会七十年性别角色变迁
1. 研究目标（Objective）

本代码旨在复现论文《Word Embeddings Track Social Group Changes Across 70 Years in China》中提出的历时性词向量分析方法，揭示中国社会七十年间（1950–2019）性别角色（男性、女性）的社会认知变迁。
三层分析框架复现：

年度词向量模型（1950-2019）：捕捉细粒度语义变迁
Analysis 1：性别效价不对称（Gender Valence Asymmetry）
Analysis 2：社会群体表征的时间演变（Temporal evolution of social group representations）
Analysis 3：通过逐年嵌入量化社会变迁（Quantifying social change through year-to-year embeddings）

2. 快速开始（Quick Start）
2.1 环境设置

Python版本：3.8+
核心依赖：
bash
pip install jieba gensim pandas numpy tqdm scikit-learn matplotlib datasets

2.2 数据准备

将1950–2019年《人民日报》按年份存储为：
text

D:/人民日报年份数据/
├── 1950.jsonl.gz
├── 1951.jsonl.gz
...
└── 2019.jsonl.gz

2.3 关键参数（严格对齐论文）
参数	                  论文设置            	本复现设置
模型架构	              Skip-gram	           sg=1
向量维度	              300	                 vector_size=300
上下文窗口	            3	                   window=3
最小词频	              >10	                 min_count=10
分词工具	              jieba + 自定义词典	  jieba + 自定义词典
迭代次数	              10                   epochs	epochs=10

2.4 完整复现流程
bash
# 1. 训练年度Skip-gram模型（1950-2019）
python train_yearly_sg300w3.py
# → 输出：models_yearly_sg300w3/{1950..2019}.model

# 2. 训练十年期模型（1950s-2010s）并生成Table 2
python train_decade_sg300w3_and_table2.py
# → 输出：models_decade_sg300w3/*.model + table2_reproduction_*.csv

# 3. 复现Analysis 1：性别效价不对称
python analysis1_gender_valence_replication.py
# → 输出：analysis1_gender_valence_replication_AggDiffMAC.csv

# 4. 复现Analysis 2：社会群体表征时间演变
python analysis2_gender_temporal_evolution.py
# → 输出：analysis2_gender_temporal_evolution.csv

# 5. 复现Analysis 3：逐年嵌入量化社会变迁
python analysis3_historical_events_replication.py
# → 输出：analysis3_yearly_stability.csv + analysis3_temporal_correlation.png

3. 核心文件说明
3.1 主要脚本文件
文件	                                                 功能	                                             关键参数
train_yearly_sg300w3.py	                              训练1950–2019共70个年度Skip-gram模型	             sg=1, dim=300, win=3, min_count=10
train_decade_sg300w3_and_table2.py	                   训练7个十年期模型 + 生成Table 2（女性Top 10特质）  同上
analysis1_gender_valence_replication.py	              复现Analysis 1（性别效价不对称）	                  AggDiffMAC计算
analysis2_gender_temporal_evolution.py	               复现Analysis 2（时间演变）	                       十年期效价对比
analysis3_historical_events_replication.py	           复现Analysis 3（逐年相关性）	                      相关性分析 + 热力图

3.2 数据文件
文件	                                                 格式	                                             用途
trait_cn.txt	                                         空格分隔文本（词 效价）	                           中文人格特质词典（含效价评分）
trait_cn_cleaned.txt	                                 空格分隔文本                    	                 清洗后的特质词典（剔除异常词）
人民日报年份数据/*.jsonl.gz	                          JSON Lines + Gzip	                                1950-2019年原始语料

3.3 输出文件
文件	                                                 生成来源	                                         内容
models_yearly_sg300w3/*.model	                        train_yearly_sg300w3.py	                          年度词向量模型
models_decade_sg300w3/*.model	                        train_decade_sg300w3_and_table2.py	               十年期词向量模型
table2_reproduction_*.csv	                            同上	                                             Table 2复现结果（女性特质Top 10）
analysis1_gender_valence_replication_AggDiffMAC.csv	  analysis1_gender_valence_replication.py	          Analysis 1详细结果
analysis2_gender_temporal_evolution.csv	              analysis2_gender_temporal_evolution.py	           Analysis 2详细结果
analysis3_yearly_stability.csv	                       analysis3_historical_events_replication.py       	Analysis 3逐年相关性
analysis3_temporal_correlation.png	                   同上	                                             相关性热力图

4. 结果示例
4.1 Analysis 1 输出
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

4.2 与论文预期对比
时期	                                           复现结果	                                        论文预期                	                                          一致性
1950s–1960s	                                    女性效价 >> 男性（0.94 > 0.53; 0.73 > 0.36）	    女性被赋予高度正面道德特质（如"贤淑"、"天真"）     	                ✅
1970s	                                          男性效价反超（0.60 < 0.81）	                     文革时期"去性别化"，强调"革命性"、"上进"等中性/男性化特质           	✅
1990s–2000s	                                    男性效价为负（-0.42, -0.10）	                    改革开放后，男性关联更多负面词（"冒失"、"邋遢"、"心狠手辣"）	        ✅
2010s	                                          两性趋近（0.32 vs 0.25）	                        当代性别话语复杂化，传统与现代特质混合	                             ✅

4.3 Analysis 2 输出
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

4.4 Analysis 3 输出
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

4.5 热力图（逐年相关性）
<img width="1442" height="569" alt="逐年词向量相关性热力图" src="https://github.com/user-attachments/assets/ebe82811-dd3a-4cac-a713-f6df2a67ca69" />








