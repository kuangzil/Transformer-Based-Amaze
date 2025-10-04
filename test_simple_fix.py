#!/usr/bin/env python3
"""
简单测试修复后的代码
"""
from maze_generator import MazeGenerator

def test_simple():
    print("=== 简单测试 ===")
    
    try:
        generator = MazeGenerator()
        print("生成器初始化成功")
        
        # 测试简单句子
        test_sentence = "The cat sat."
        print(f"测试句子: {test_sentence}")
        
        result = generator.generate_full_maze(test_sentence)
        print("生成完成!")
        
        print("结果:")
        for pair in result['maze_pairs']:
            print(f"  Position {pair['position']}: {pair['correct']} vs {pair['distractor']}")
        
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple()
