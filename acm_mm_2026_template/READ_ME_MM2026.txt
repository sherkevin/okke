ACM MM 2026 — 官方说明与本地文件指引
=====================================

一、官方网站（投稿标准与流程）
----------------------------
作者说明（必读）:
  https://2026.acmmm.org/site/author-instructions.html
本地副本: ACM_MM_2026_author-instructions.html（与本文件同目录）

ACM 统一论文模板说明与下载页（LaTeX / Word）:
  https://www.acm.org/publications/proceedings-template

LaTeX 类文件用户指南 PDF（ACM 官方）:
  https://portalparts.acm.org/hippo/latex_templates/acmart.pdf
本地副本: acmart-user-guide.pdf

二、ACM MM 2026 对 LaTeX 的明确要求（摘自作者说明页）
----------------------------------------------------
1. 须使用 ACM Article Template；提交 PDF。
2. LaTeX 作者请使用「双栏」会议稿模板中的 sample-sigconf-authordraft。
3. 投稿与审稿阶段，建议将文档类改为（匿名审稿）:
   \documentclass[sigconf, screen, review, anonymous]{acmart}
   （官方说明中给出的示例；请在 sample-sigconf-authordraft.tex 中替换原 \documentclass 行。）
4. 须包含 ACM CCS Concepts 与 Keywords。
5. 通过 OpenReview 投稿；各 track 截稿见各自 CFP。
6. 主会等技术轨常见篇幅：正文最多 8 页 + 参考文献最多额外 2 页（以官方表格为准）。
7. 除 Reproducibility 轨外，需双盲；作者说明页列有去标识化要求。

三、本目录已下载 / 生成的内容
----------------------------
- acmart-ctan.zip
    CTAN 上的 acmart 宏包（与 ACM 官方 acmart 同源发行）。本机访问 ACM
    portal 的 acmart-primary.zip 若返回 403，可用此 zip 或 TeX Live 中的
    acmart 包。

- acmart\acmart\
    已解压的宏包根目录。其中已运行 acmart.ins 生成 acmart.cls；
    samples\ 下已运行 latex samples.ins，已生成 sample-sigconf-authordraft.tex
    等示例源文件。

编译示例（在 samples 目录下）:
  pdflatex sample-sigconf-authordraft.tex
  bibtex sample-sigconf-authordraft
  pdflatex sample-sigconf-authordraft.tex
  pdflatex sample-sigconf-authordraft.tex

（请先将 \documentclass 按上文改为 MM 2026 推荐的 review+anonymous 选项。）

四、其他官方参考链接
--------------------
- ACM 参考文献格式: https://www.acm.org/publications/authors/reference-formatting
- BibTeX 说明: https://www.acm.org/publications/authors/bibtex-formatting
- LaTeX 允许使用的宏包白名单: https://www.acm.org/publications/taps/whitelist-of-latex-packages
- CCS 概念生成: https://dl.acm.org/ccs/ccs.cfm

（说明文件生成日期以文件系统为准；截稿与政策请以 2026.acmmm.org 最新页面为准。）
