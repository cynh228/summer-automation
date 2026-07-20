import random #导入随机模块
secret = random.randint(1,100) #1-100随机一个数
print("=== 欢迎来到猜数字游戏！===")
print("我已经想好了 1 到 100 之间的一个数。")

while True:
    # 获取玩家输入（注意：input 返回的是字符串，要用 int() 转成数字）
    guess = int(input("请输入你猜的数字: "))
    
    # 3. 条件判断（if / elif / else）
    if guess < secret:
        print("太小了，再大一点！")
    elif guess > secret:
        print("太大了，再小一点！")
    else:
        print(f"恭喜你！猜对了！答案就是 {secret} ！")
        break  # 猜对了，跳出循环，游戏结束

print("游戏结束，感谢游玩！")