"""
RAG评估快速启动脚本
从项��根目录运行评估，无需切换到evaluation文件夹

使用方法:
    # 测试API连接
    python run_evaluation.py test-api --provider glm

    # 运行完整评估
    python run_evaluation.py evaluate --provider glm --num-questions 50

    # 查看帮助
    python run_evaluation.py --help
"""

import sys
import argparse
from pathlib import Path

# 添加项目根路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入evaluation模块
from evaluation.scripts.test_api_connection import test_api_connection


def main():
    parser = argparse.ArgumentParser(
        description='RAG评估快速启动',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试API连接
  python run_evaluation.py test-api --provider glm --api-key "your-key"

  # 运行快速评估（10个问题）
  python run_evaluation.py evaluate --provider glm --num-questions 10

  # 运行标准评估（50个问题）
  python run_evaluation.py evaluate --provider glm --num-questions 50

  # 使用环境变量中的API Key
  export GLM_API_KEY="your-key"
  python run_evaluation.py evaluate --provider glm
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # test-api命令
    test_api_parser = subparsers.add_parser(
        'test-api',
        help='测试API连接'
    )
    test_api_parser.add_argument(
        '--provider',
        choices=['glm', 'openai', 'qwen', 'azure'],
        default='glm',
        help='LLM提供商 (默认: glm)'
    )
    test_api_parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API密钥'
    )
    test_api_parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型名称'
    )

    # evaluate命令
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='运行RAGAS评估'
    )
    eval_parser.add_argument(
        '--provider',
        choices=['glm', 'openai', 'qwen', 'azure'],
        default='glm',
        help='LLM提供商 (默认: glm)'
    )
    eval_parser.add_argument(
        '--num-questions',
        type=int,
        default=50,
        help='测试问题数量 (默认: 50)'
    )
    eval_parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API密钥'
    )
    eval_parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型名称'
    )
    eval_parser.add_argument(
        '--use-existing-data',
        action='store_true',
        help='使用已有测试数据'
    )
    eval_parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='批处理大小'
    )

    args = parser.parse_args()

    if args.command == 'test-api':
        # 测试API连接
        print("\n🔍 测试API连接...\n")
        success = test_api_connection(
            provider=args.provider,
            api_key=args.api_key,
            model=args.model
        )
        sys.exit(0 if success else 1)

    elif args.command == 'evaluate':
        # 运行评估
        print(f"\n🚀 运行RAGAS评估 (Provider: {args.provider})\n")

        # 构建评估命令
        import subprocess

        cmd = [
            sys.executable,
            'evaluation/scripts/run_full_ragas_evaluation.py',
            '--provider', args.provider,
            '--num-questions', str(args.num_questions)
        ]

        if args.api_key:
            cmd.extend(['--api-key', args.api_key])
        if args.model:
            cmd.extend(['--model', args.model])
        if args.use_existing_data:
            cmd.append('--use-existing-data')
        if args.batch_size != 5:
            cmd.extend(['--batch-size', str(args.batch_size)])

        # 运行评估
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
