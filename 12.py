#写一个汉诺塔递归函数
def hanoi(n, source, target, auxiliary):
    if n > 0:
        # 将 n-1 个盘子从源柱移动到辅助柱
        hanoi(n-1, source, auxiliary, target)

        # 将第 n 个盘子从源柱移动到目标柱
        print("Move disk", n, "from", source, "to", target)

        # 将 n-1 个盘子从辅助柱移动到目标柱
        hanoi(n-1, auxiliary, target, source)

# 测试
hanoi(3, 'A', 'C', 'B')
