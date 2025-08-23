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
然后写一个脚本，将数据集转化为这种结构
```plaintext
dataset
    metadata.json
    select_oblige
        fem_kuk_00001.ogg
    senren_banka
        akh108_001.ogg
```
而元数据`metadata.json`内容应类似这样：对于每一个音频，都应该在元数据有一个对应的条目
```json
{
    "select_oblige/kuk/fem_kuk_00001.ogg": {
        "text": "XXXX",
        "positive_prompt": ["sex:female"],
        "negative_prompt": ["sex:male"]
    },
    "senren_banka/akh108_001.ogg": {
        "text": "XXXX",
        "positive_prompt": ["sex:female"],
        "negative_prompt": ["sex:male"]
    }
}
```

#### 从 Artemis 游戏中提取数据集
1. 使用[GARbro](https://github.com/crskycode/GARbro)从游戏目录的`xxx.pfs`文件提取出`sound/vo`和`script`文件夹，分别保存到`/path/to/game/sound/vo`和`/path/to/game/script`
2. 运行`python extract_artemis.py /path/to/game/script /path/to/game/sound/vo /path/to/artemis_pre_dataset`，它会输出一个角色ID对应的角色名次数
3. 仿照`examples/select_oblige_c2t.json`和`examples/aikotoba_sss_c2t.json`，根据第二步的输出，写你要提取的游戏的角色ID-角色名映射表，保存到`/path/to/artemis_c2t.json`
4. 运行`python convert_artemis_to_dataset.py /path/to/artemis_pre_dataset /path/to/artemis_c2t.json /path/to/artemis_dataset -p source:<游戏名> -p <其他你想在所有对话加上的标签> -n <你想在所有对话加上的负面标签>`
5. 你的数据集应该已经在`/path/to/artemis_dataset`了，元数据文件在`/path/to/artemis_dataset/metadata.json`

#### 从 Kirikiri Z 游戏中提取数据集
1. 使用[GARbro](https://github.com/crskycode/GARbro)分别从游戏目录的`data.xp3`文件和`voice.xp3`提取出`scn`和根目录文件夹，分别保存到`/path/to/game/script`和`/path/to/game/voice`
2. 运行`python extract_kirikiriz.py /path/to/game/script /path/to/game/voice /path/to/kirikiriz_pre_dataset`，它会输出所有对话出现的角色名
3. 仿照`examples/senren_banka_c2t.json`，根据第二步的输出，写你要提取的游戏的角色ID-角色名映射表，保存到`/path/to/kirikiriz_c2t.json`
4. 运行`python convert_kirikiriz_to_dataset.py /path/to/kirikiriz_pre_dataset /path/to/kirikiriz_c2t.json /path/to/kirikiriz_dataset -p source:<游戏名> -p <其他你想在所有对话加上的标签> -n <你想在所有对话加上的负面标签>`
5. 你的数据集应该已经在`/path/to/kirikiriz_dataset`了，元数据文件在`/path/to/kirikiriz_dataset/metadata.json`

### 可选: 对数据集进行提示词增强
我们都知道，有一些标签是不可能同时存在的，比如`sex:male`和`sex:female`就不可能同时出现在一个正面提示里，所以我们可以定义一堆标签互斥组，类似这样
```python
[
    {"sex:male", "sex:female"},
    {"age:children", "age:teenager", "age:adult", "age:old"},
    {"character:白上フブキ", "character:夏色まつり"}  # 举这两个例子是因为我喜欢看
]
```
对于一个音频的元数据，我们遍历每一个标签互斥组（类型：集合），然后检测元数据的正面标签是否包含且仅包含了这个标签互斥组中的一个标签，比如
```python
positive_prompt: set[str]
negative_prompt: set[str]
groups = [{"age:children", "age:teenager", "age:adult", "age:old"}, ...]
for group in groups:
    if len(intersection := positive_prompt & group) == 1:  # 为什么不是只要检测到包含就全部加入负面提示呢？因为可能有些用户想要同时指定两个冲突的标签；鬼知道他们是怎么想的
        negative_prompt.update(group - intersection)  # 将该组其余所有的互斥标签加入负面提示
```
我写了一个脚本来节省自己造轮子的麻烦，你可以通过运行
```bash
python augment_prompt_with_exclusion.py /path/to/dataset/metadata.json /path/to/dataset/metadata_augment.json /path/to/mutually_exclusive_groups.json
```
来实现提示词增强

### 可选: 合并数据集
如果你有若干个数据集，像这样
```plaintext
datasets
    shirakami_fubuki
        metadata.json
        ytb_live01_001.oog
        ytb_live01_002.oog
        ...
    natsuiro_matsuri
        metadata.json
        ytb_live01_001.oog
        ytb_live01_002.oog
        ...
```
你可以通过将不同数据集放在一个文件夹，比如`/path/to/datasets`，然后运行
```bash
python merge_datasets.py /path/to/datasets/shirakami_fubuki/metadata.json /path/to/datasets/natsuiro_matsuri/metadata.json -o /path/to/datasets/metadata.json
```
将其元数据合并到`/path/to/datasets/metadata.json`；此外，你可以将原来的`/path/to/datasets/shirakami_fubuki/metadata.json`和`/path/to/datasets/natsuiro_matsuri/metadata.json`删除

### 可选: 切割数据集
你可以在训练前将数据集切割为训练集和验证集；这只需要将`metadata.json`拆分为`train.json`和`val.json`，你可以通过运行
```bash
python split_dataset.py /path/to/dataset/metadata.json train.json:9 val.json:1
```
这样，你的`/path/to/dataset`目录应出现占总数据九成比例`train.json`和一成比例的`val.json`。

### 训练分词器
```bash
python train_tokenizer.py /path/to/ckpt -t /path/to/dataset/train.json -v /path/to/dataset/val.json
```

### 初始化检查点
```bash
python list_tags_from_datasets.py /path/to/dataset/train.json -o /path/to/tags.txt
python init_checkpoint.py /path/to/ckpt -t /path/to/tags.txt
```

### 将通用数据集转化为快速训练数据集
由于直接训练时加载音频数据十分缓慢，不能发挥 GPU 训练的快的优势，所以这里我们采用：将训练数据预处理缓存在硬盘，在训练时直接加载而无需处理的方法，加快数据加载速度。

这样的数据集只适用于训练以下超参完全相同的检查点：
- tokenizer
- tag_label_encoder（但是如果是在原始标签后又新增标签的没事，比如说，``{"[UNK]": 0, "tag": 1, "new": 2}`可以兼容`{"[UNK]": 0, "tag": 1}`的标签编码器）
- sample_rate
- fft_length
- frame_length
- hop_length
- win_length
- num_mels

将通用数据集转化为快速训练数据集，包含几个步骤：
- 文本：用分词器编码为文本序列
- 正面、负面提示词：用标签编码器转换为提示词序列
- 音频：转换为梅尔频谱、音高和能量

考虑到兼容两种加载方式（加载原始通用数据集和快速训练数据集）的代码很难维护（见[tkaimidi](https://github.com/thiliapr/tkaimidi)一直没维护），所以这里强制要求必须转换为快速训练数据集。你可以运行
```bash
python prepare_fast_dataset.py /path/to/dataset/metadata.json /path/to/ckpt /path/to/fast_dataset
```
新的元数据将会在`/path/to/fast_dataset/metadata.json`

### 训练模型
```bash
python train_tktts.py <num_epochs> /path/to/ckpt -t /path/to/dataset/train.json -v /path/to/dataset/val.json
```
将`<num_epochs>`替换为实际的你想训练的轮数  
> [!TIP]
> 你可以在训练后运行`python show_scales.py /path/to/ckpt`来看看每层的缩放因子，按数据流向排序

### 生成
```bash
python list_tags_from_ckpt.py /path/to/ckpt  # 列出所有标签
python generate.py /path/to/ckpt output.wav ホロライブ所属のバーチャルyoutuber、夏色まつりだよっ！！ -p character:夏色まつり -p source:youtube -p sex:female -p age:teenager -n sex:male
```

### 注意事项
- 请在命令行输入`python3 file.py --help`获得帮助

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