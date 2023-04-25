# 基础模型操作

## 生成最小mvp范例
```
# 在当前文件夹下创建最小mvp代码范例
antgo create mvp --name=lsp
```
## 模型训练/测试/导出
```
# step1: 模型训练
# 单机1卡训练(可以自定义使用第几块卡 --gpu-id=0)
python3 ./lsp/main.py --exp=lsp --gpu-id=0 --no-validate --process=train
# 单机CPU训练（仅用于调试）
python3 ./lsp/main.py --exp=lsp --gpu-id=-1 --no-validate --process=train
# 单机多卡训练（4卡运行）
bash launch.sh ./lsp/main.py 4 --exp=lsp --no-validate --process=train

# step2: 模型测试
python3 ./lsp/main.py --exp=lsp --checkpoint='' --gpu-id=0 --process=test


# step3: 模型导出
python3 ./lsp/main.py --exp=lsp --checkpoint='' --process=export
```

# 项目管理操作
# humantracking
