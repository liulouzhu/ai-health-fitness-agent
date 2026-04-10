#!/usr/bin/env python3
"""使用 LangGraph 的 get_graph() 生成工作流程图

生成后的 Mermaid 图可用于 README 或 Mermaid Live Editor 渲染。

Usage:
    python generate_graph_diagram.py                    # 输出 Mermaid 语法
    python generate_graph_diagram.py -f png -o graph.png  # 输出 PNG
    python generate_graph_diagram.py -f svg -o graph.svg  # 输出 SVG
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from agent.graph import create_workflow


def main():
    parser = argparse.ArgumentParser(
        description="生成 LangGraph 工作流图",
        epilog="""
工作流节点说明：
  check_profile     - 检查用户档案完整性
  init_daily_stats  - 初始化每日统计
  classify_intent   - LLM 意图分类
  food_generate     - 食物分析与候选生成
  workout_generate  - 运动指导与候选生成
  stats_node        - 今日统计查询
  recipe_node       - 食谱推荐
  profile_node      - 档案更新
  general_node      - 通用对话
  confirm_node      - 展示待确认内容
  confirm_recovery  - 处理用户确认回复
  commit_node       - 将记录写入 daily_stats
  food_branch       - fan-out 中的 food 分支
  workout_branch    - fan-out 中的 workout 分支
  stats_branch      - fan-out 中的 stats 分支
  multi_join_node   - 多意图结果合并
        """
    )
    parser.add_argument(
        "--format", "-f",
        choices=["mermaid", "graphviz", "png", "svg"],
        default="mermaid",
        help="输出格式: mermaid (默认), png, svg"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="输出文件路径"
    )
    args = parser.parse_args()

    app = create_workflow()
    graph = app.get_graph()

    if args.format == "mermaid":
        # 输出 Mermaid 语法
        mermaid_code = graph.draw_mermaid()

        if args.output:
            args.output.write_text(mermaid_code, encoding="utf-8")
            print(f"已生成 Mermaid 图: {args.output}")
        else:
            print(mermaid_code)

    elif args.format == "graphviz":
        # 输出 Graphviz DOT 语法
        try:
            dot_code = graph.draw_graphviz()
            if args.output:
                args.output.write_text(dot_code, encoding="utf-8")
                print(f"已生成 Graphviz 图: {args.output}")
            else:
                print(dot_code)
        except Exception as e:
            print(f"生成 Graphviz 失败: {e}")

    elif args.format == "png":
        # 输出 PNG 图片
        try:
            png_data = graph.draw_mermaid_png()
            if args.output:
                args.output.write_bytes(png_data)
                print(f"已生成 PNG 图: {args.output}")
            else:
                print("PNG data length:", len(png_data), "bytes")
        except Exception as e:
            print(f"生成 PNG 失败: {e}")
            print("尝试使用 --format=mermaid 然后用 Mermaid Live Editor 转换")

    elif args.format == "svg":
        # 输出 SVG
        try:
            svg_data = graph.draw_mermaid_svg()
            if args.output:
                args.output.write_text(svg_data, encoding="utf-8")
                print(f"已生成 SVG 图: {args.output}")
            else:
                print(svg_data)
        except Exception as e:
            print(f"生成 SVG 失败: {e}")


if __name__ == "__main__":
    main()
