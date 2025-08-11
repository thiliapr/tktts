# Tk TTS
## 简介
这是一个自由的语音生成软件，主要使用[librosa](https://librosa.org/)来解析音频，[pytorch](https://github.com/pytorch/pytorch/)来机器学习，旨在通过深度学习技术生成音乐作品。

## License
![GNU AGPL Version 3 Logo](https://www.gnu.org/graphics/agplv3-with-text-162x68.png)

TkTTS 是自由软件，遵循`Affero GNU 通用公共许可证第 3 版或任何后续版本`。你可以自由地使用、修改和分发该软件，但不提供任何明示或暗示的担保。有关详细信息，请参见 [Affero GNU 通用公共许可证](https://www.gnu.org/licenses/agpl-3.0.html)。

## 使用示例
这里演示的是大致流程，实际可能需要调整，不过一般照着这个来就行了

### 安装依赖
#### CPU 用户
```bash
pip install -r requirements.txt
```

#### CUDA 用户（Nvidia 显卡）
```bash
nvidia-smi  # 查看 CUDA 版本
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

> [!TIP]
> 根据`nvidia-smi`的输出`CUDA Version`把`cu128`换成你自己的 CUDA 版本，比如输出`CUDA Version: 12.1`就把`cu128`替换为`cu126`  
> 具体来说，PyTorch 的CUDA是向下兼容的，所以选择时只需要选择比自己的 CUDA 版本小一点的版本就行了。  
> 比如 PyTorch 提供了三个版本: `12.6, 12.8, 12.9`，然后你的 CUDA 版本是`12.7`，那么就选择`12.8`（因为官方提供的`12.6` < 你的`12.7` < 官方提供的`12.8`）

### 准备数据集
准备一个数据集，比如我用游戏提取脚本提取游戏语音作为数据集，这里演示使用 The LJ Speech Dataset 数据集

#### 预处理一般数据集
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2  # https://keithito.com/LJ-Speech-Dataset/
```
然后写一个脚本，将数据集转化为这种格式（自食其力，或者发issue叫我添加功能）
```json
[{"tags":["character:白上フブキ","sex:female","age:teenager"],"text":"Hi Friends!"}]
```

#### 从 Artemis 游戏中提取数据集
1. 使用[GARbro](https://github.com/crskycode/GARbro)从游戏目录的`xxx.pfs`文件提取出`sound/vo`和`script`文件夹，分别保存到`/path/to/game/sound/vo`和`/path/to/game/script`
2. 运行`python extract_artemis.py /path/to/game/script /path/to/game/sound/vo /path/to/artemis_pre_dataset`，它会输出一个角色ID对应的角色名次数
3. 仿照`examples/select_oblige_c2t.json`和`examples/aikotoba_sss_c2t.json`，根据第二步的输出，写你要提取的游戏的角色ID-角色名映射表，保存到`/path/to/artemis_c2t.json`
4. 运行`python convert_artemis_to_dataset.py /path/to/artemis_pre_dataset /path/to/artemis_c2t.json /path/to/artemis_dataset -t source:<游戏名> -t <其他你想在所有对话加上的标签>`
5. 你的数据集应该已经在`/path/to/artemis_dataset`了

#### 从 Kirikiri Z 游戏中提取数据集
1. 使用[GARbro](https://github.com/crskycode/GARbro)分别从游戏目录的`data.xp3`文件和`voice.xp3`提取出`scn`和根目录文件夹，分别保存到`/path/to/game/script`和`/path/to/game/voice`
2. 运行`python extract_kirikiriz.py /path/to/game/script /path/to/game/voice /path/to/kirikiriz_pre_dataset`，它会输出所有对话出现的角色名
3. 仿照`examples/senren_banka_c2t.json`，根据第二步的输出，写你要提取的游戏的角色ID-角色名映射表，保存到`/path/to/kirikiriz_c2t.json`
4. 运行`python convert_kirikiriz_to_dataset.py /path/to/kirikiriz_pre_dataset /path/to/kirikiriz_c2t.json /path/to/kirikiriz_dataset -t source:<游戏名> -t <其他你想在所有对话加上的标签>`
5. 你的数据集应该已经在`/path/to/kirikiriz_dataset`了

### 训练分词器
```bash
python train_tokenizer.py /path/to/ckpt -t /path/to/train_dataset -v /path/to/val_dataset
```

### 初始化检查点
```bash
python init_checkpoint.py /path/to/ckpt
```

### 训练模型
```bash
python train_tktts.py <num_epochs> /path/to/ckpt -t /path/to/train_dataset -v /path/to/val_dataset
```
将`<num_epochs>`替换为实际的你想训练的轮数  
> [!TIP]
> 你可以在训练途中或训练后运行`python show_scales.py /path/to/ckpt`来看看每层的缩放因子，按数据流向排序

### 生成
```bash
python list_tags.py  # 列出所有标签
python generate.py ckpt output.wav "[character:夏色まつり][sex:female][age:teenager]ホロライブ所属のバーチャルyoutuber、夏色まつりだよっ！！"  # 生成语音（格式: `[标签1][标签2]...[标签N]文本`）
```

### 注意事项
- 请在命令行输入`python3 file.py --help`获得帮助

## 模型结构
以下以纯文本形式展示了模型的结构，省略了 padding_mask 和 kv_cache 部分。之所以是纯文本，是因为我不会画图ww
```plaintext
Module Linear(x), GeLU(x), ScaleNorm(x), MultiheadAttentionWithRoPE(q, kv), Sequential(x);
Module EncoderLayer:
    init:
        .attn_scale = .ff_scale=0
        .attn: MultiheadAttentionWithRoPE
        .ff = Sequential(Linear, GeLU, Linear)
        .attn_norm, .ff_norm: ScaleNorm
    forward(x):
        norm_x = attn_norm(x)
        x = x + .attn(norm_x, norm_x) * .attn_scale
        x = x + .ff(.ff_norm(x)) * .ff_scale
Module DecoderLayer:
    init:
        .sa_scale = .ca_scale = .ff_scale=0
        .sa, .ca: MultiheadAttentionWithRoPE
        .ff = Sequential(Linear, GeLU, Linear)
        .sa_norm, .ca_norm, .ff_norm: ScaleNorm
    forward(target, memory):
        norm_target = sa_norm(target)
        x = target + .sa(norm_target, norm_target, is_causal=True) * .sa_scale
        x = x + .ca(.ca_norm(x), memory) * .ca_scale
        x = x + .ff(.ff_norm(x)) * .ff_scale
Module TkTTS:
    init:
        .embedding: Embedding
        .audio_proj, .audio_pred, .stop_pred: Linear
        .encoder = [EncoderLayer for _ in range(N)]
        .decoder = [DecoderLayer for _ in range(N)]
    forward(source, target):
        memory = .embedding(source)
        target = .audio_proj(target)
        for layer in .encoder:
            memory = layer(memory)
        for layer in .decoder:
            target = layer(target, memory)
        return .audio_pred(target), .stop_pred(target)[..., 0]
```

## 文档
文档是不可能写的，这辈子都不可能写的。经验表明，写了文档只会变成“代码一天一天改，文档一年不会动”的局面，反而误导人。

所以我真心推荐：有什么事直接看代码（代码的注释和函数的文档还是会更新的），或者复制代码问ai去吧（记得带上下文）。

## 贡献与开发
欢迎提出问题、改进或贡献代码。如果有任何问题或建议，您可以在 GitHub 上提 Issues，或者直接通过电子邮件联系开发者。

## 联系信息
如有任何问题或建议，请联系项目维护者 thiliapr。
- Email: thiliapr@tutanota.com

# 无关软件本身的广告
## Join the Blue Ribbon Online Free Speech Campaign!
![Blue Ribbon Campaign Logo](https://www.eff.org/files/brstrip.gif)

支持[Blue Ribbon Online 言论自由运动](https://www.eff.org/pages/blue-ribbon-campaign)！  
你可以通过向其[捐款](https://supporters.eff.org/donate)以表示支持。

## 支持自由软件运动
为什么要自由软件: [GNU 宣言](https://www.gnu.org/gnu/manifesto.html)

你可以通过以下方式支持自由软件运动:
- 向非自由程序或在线敌服务说不，哪怕只有一次，也会帮助自由软件。不和其他人使用它们会帮助更大。进一步，如果你告诉人们这是在捍卫自己的自由，那么帮助就更显著了。
- [帮助 GNU 工程和自由软件运动](https://www.gnu.org/help/help.html)
- [向 FSF 捐款](https://www.fsf.org/about/ways-to-donate/)